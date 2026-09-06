from openai import OpenAI
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from . import config
import json


class RAGAnswer(BaseModel):
    answer: str = Field(description="The answer to the question based on the provided context.")
    confidence: float = Field(default=0.0, description="Confidence score (0.0-1.0) in the answer. Higher is better.")
    sources: List[str] = Field(default_factory=list, description="List of document chunks used to formulate the answer.")
    reasoning: Optional[str] = Field(default=None, description="Optional reasoning for the answer or why structured parse failed.")


class RAGPipeline:
    """End-to-end RAG pipeline: retrieve → re-rank → generate."""

    def __init__(
        self,
        vector_store,
        hybrid_retriever,
        reranker,
        llm_model: str = None,
        strategy: str = "recursive",
        enable_tools: bool = False,
    ):
        self.vector_store = vector_store
        self.hybrid_retriever = hybrid_retriever
        self.reranker = reranker
        self.strategy = strategy
        self.enable_tools = enable_tools

        # Initialize LLM model name
        # Note: The litellm proxy is started with: litellm --model vertex_ai/gemini-2.5-flash
        # Use gemini-primary as the model name when talking to the proxy via OpenAI client
        if llm_model is None:
            llm_model = "gemini-primary"

        self.llm_model = llm_model
        self.proxy_url = config.LITELLM_PROXY_URL

        # Initialize OpenAI client to talk to LiteLLM proxy
        # The proxy accepts OpenAI-compatible API format
        self.client = OpenAI(
            api_key="anything",  # LiteLLM proxy handles auth via Vertex AI creds
            base_url=self.proxy_url
        )

    def _build_context(self, reranked_results: List[Dict[str, Any]]) -> str:
        """Build context string from re-ranked retrieval results."""
        context_parts = []
        for i, result in enumerate(reranked_results):
            doc_text = result.get("document", "")
            metadata = result.get("metadata", {})
            source = metadata.get("source", "unknown")
            chunk_idx = metadata.get("chunk_idx", i)
            context_parts.append(
                f"[Chunk {chunk_idx} from {source}]\n{doc_text}"
            )
        return "\n\n".join(context_parts)

    def _construct_prompt(self, question: str, context: str) -> str:
        """Construct the prompt for the LLM."""
        prompt = f"""You are a knowledgeable assistant answering questions about the Ramayana.

Use the provided context from the Ramayana document to answer the user's question.
If the answer cannot be found in the context, say "I cannot find this information in the Ramayana document."

Context:
{context}

Question: {question}

Answer:"""
        return prompt

    def get_document_stats(self, strategy: str) -> Dict[str, Any]:
        """Tool function to get vector store statistics for a strategy."""
        stats = self.vector_store.get_collection_stats()
        # Find the collection that matches the strategy
        # Collection names in Chroma are prefixed with 'ramayana_'
        collection_name = f"ramayana_{strategy}"
        if collection_name in stats:
            res = stats[collection_name]
            return {
                "strategy": strategy,
                "chunk_count": res.get("document_count", 0),
                "collection_size": res.get("document_count", 0), # Simplified for demo
            }
        return {"error": f"Strategy {strategy} not found in vector store."}

    def _get_tools_schema(self) -> List[Dict[str, Any]]:
        """Return the tool schema for the LLM."""
        return [
            {
                "type": "function",
                "function": {
                    "name": "get_document_stats",
                    "description": "Get chunk count and collection size for a specific chunking strategy.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "strategy": {
                                "type": "string",
                                "enum": ["fixed", "recursive", "semantic", "structural"],
                                "description": "The chunking strategy name."
                            }
                        },
                        "required": ["strategy"]
                    }
                }
            }
        ]

    def _query_llm(self, prompt: str, is_retry: bool = False, messages: List[Dict[str, Any]] = None) -> RAGAnswer:
        """Query the LiteLLM proxy for a completion via OpenAI-compatible API, requesting structured output."""
        schema_instruction = (
            "\n\nRespond ONLY with valid JSON matching this exact schema (no extra text, "
            "no markdown code fences):\n"
            f"{RAGAnswer.model_json_schema()}\n"
        )

        if messages is None:
            messages = [{"role": "user", "content": prompt + schema_instruction}]

        if is_retry:
            messages.append({"role": "user", "content": "Your previous response was not valid JSON. Please return ONLY valid JSON matching the schema." + schema_instruction})

        kwargs = {
            "model": self.llm_model,
            "messages": messages,
            "temperature": config.TEMPERATURE,
            "max_tokens": config.MAX_CONTEXT_TOKENS,
        }

        if self.enable_tools:
            kwargs["tools"] = self._get_tools_schema()
            kwargs["tool_choice"] = "auto"

        try:
            response = self.client.chat.completions.create(**kwargs)
            message = response.choices[0].message

            # Handle Tool Calls
            if message.tool_calls:
                messages.append(message)
                for tool_call in message.tool_calls:
                    if tool_call.function.name == "get_document_stats":
                        args = json.loads(tool_call.function.arguments)
                        result = self.get_document_stats(args.get("strategy", self.strategy))
                        messages.append({
                            "tool_call_id": tool_call.id,
                            "role": "tool",
                            "name": "get_document_stats",
                            "content": json.dumps(result)
                        })
                
                # Follow-up after tool execution
                # Note: We must ensure the final response is structured
                return self._query_llm(prompt, messages=messages)

            # Request structured output via a final completion if tool calls are done
            # or if tools were not called. 
            # Note: LiteLLM response_format/functions can be tricky combined with tools.
            # Using functions for structured output as requested.
            
            _schema = RAGAnswer.model_json_schema()
            structured_kwargs = {
                "model": self.llm_model,
                "messages": messages,
                "temperature": config.TEMPERATURE,
                "max_tokens": config.MAX_CONTEXT_TOKENS,
                "response_format": {"type": "json_object"},
            }
            
            final_response = self.client.chat.completions.create(**structured_kwargs)
            final_message = final_response.choices[0].message
            answer_content = final_message.content
            if answer_content:
                return RAGAnswer(**json.loads(answer_content))
            
            return RAGAnswer(answer="", confidence=0.0, reasoning="Empty response from LLM")

        except json.JSONDecodeError as e:
            if not is_retry:
                # Retry once with an explicit instruction
                print(f"[WARNING] JSON decode error: {e}. Retrying with explicit instruction.")
                return self._query_llm(prompt, is_retry=True)
            else:
                # Gracefully degrade after retry failure
                return RAGAnswer(
                    answer="I cannot provide a structured answer at this time.",
                    confidence=0.0,
                    reasoning=f"Structured parse failed after retry: {e}"
                )
        except Exception as e:
            error_msg = str(e)
            if "default credentials" in error_msg.lower() or "google auth" in error_msg.lower():
                error_msg = (
                    "LLM proxy credential error. "
                    "Ensure the LiteLLM proxy is running at: " + self.proxy_url +
                    "\n"
                    "The proxy should be configured with gemini-primary model "
                    "from litellm_config.yaml. See: https://litellm.ai/docs/proxy"
                )
            return RAGAnswer(
                answer=f"Error querying LLM: {error_msg}",
                confidence=0.0,
                reasoning=f"LLM query failed: {error_msg}"
            )

    def ask(self, question: str, top_k: int = None) -> RAGAnswer:
        """Answer a question using the RAG pipeline."""
        if top_k is None:
            top_k = config.RERANK_TOP_K

        # Step 1: Hybrid retrieval
        hybrid_results = self.hybrid_retriever.retrieve(
            question, self.strategy
        )

        # Step 2: Re-ranking
        reranked = self.reranker.rerank(question, hybrid_results, top_k=top_k)

        # Step 3: Build context
        context = self._build_context(reranked)

        # Step 4: Query LLM
        prompt = self._construct_prompt(question, context)
        rag_answer = self._query_llm(prompt)

        # Compile sources for the RAGAnswer object
        sources = []
        for result in reranked:
            meta = result.get("metadata", {})
            chunk_idx = meta.get("chunk_idx", -1)
            source_name = meta.get("source", "unknown")
            src = f"Chunk {chunk_idx} from {source_name}"
            sources.append(src)
        rag_answer.sources = sources # Update the RAGAnswer object with actual sources

        # Return the RAGAnswer object directly
        return rag_answer

    def ask_stream(self, question: str, top_k: int = None):
        """Answer a question with streaming LLM output.

        Yields text chunks as they arrive from the LLM.
        """
        # Run the same pipeline but with streaming
        hybrid_results = self.hybrid_retriever.retrieve(question, self.strategy)
        reranked = self.reranker.rerank(question, hybrid_results, top_k=top_k)
        context = self._build_context(reranked)
        prompt = self._construct_prompt(question, context)

        try:
            response = self.client.chat.completions.create(
                model=self.llm_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=config.TEMPERATURE,
                max_tokens=config.MAX_CONTEXT_TOKENS,
                stream=True,
            )
            for chunk in response:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        except Exception as e:
            yield f"Error: {str(e)}"

