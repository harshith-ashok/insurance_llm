import os
import openai
import json
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)


class DecisionEngine:
    def __init__(self):
        self.openai_client = openai.AsyncOpenAI(
            api_key=os.getenv("OPENAI_API_KEY"))

    async def evaluate_decision(self, question: str, relevant_clauses: List[Dict[str, Any]]) -> Dict[str, Any]:
        try:
            if not relevant_clauses:
                return {
                    "answer": "No relevant information found in the document.",
                    "rationale": "The document does not contain information relevant to the question.",
                    "relevant_clauses": [],
                    "confidence": 0.0
                }

            context = self._build_context(relevant_clauses)
            prompt = self._build_prompt(question, context)

            response = await self._get_gpt_response(prompt)
            parsed_response = self._parse_response(response)

            return {
                "answer": parsed_response.get("answer", "Unable to determine answer."),
                "rationale": parsed_response.get("rationale", "Insufficient information to provide rationale."),
                "relevant_clauses": relevant_clauses[:3],
                "confidence": parsed_response.get("confidence", 0.5)
            }

        except Exception as e:
            logger.error(f"Error in decision evaluation: {str(e)}")
            return {
                "answer": "Error processing the question.",
                "rationale": f"Technical error: {str(e)}",
                "relevant_clauses": relevant_clauses[:3] if relevant_clauses else [],
                "confidence": 0.0
            }

    def _build_context(self, clauses: List[Dict[str, Any]]) -> str:
        context_parts = []
        for i, clause in enumerate(clauses[:5]):
            context_parts.append(
                f"Clause {i+1} (Type: {clause.get('type', 'general')}): {clause['content']}")
        return "\n\n".join(context_parts)

    def _build_prompt(self, question: str, context: str) -> str:
        return f"""You are an expert insurance and legal document analyzer. Based on the provided document clauses, answer the question with a structured response.

Document Clauses:
{context}

Question: {question}

Please provide a JSON response with the following structure:
{{
    "answer": "Direct answer to the question based on the clauses",
    "rationale": "Detailed explanation of how the clauses support the answer",
    "confidence": 0.0-1.0
}}

Focus on specific details like:
- Grace periods
- Waiting periods  
- Coverage limits
- Exclusions
- Conditions
- Specific medical procedures mentioned
- Policy terms and definitions

If the clauses don't contain relevant information, state that clearly in the answer."""

    async def _get_gpt_response(self, prompt: str) -> str:
        try:
            response = await self.openai_client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are an expert document analyzer that provides structured JSON responses."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=1000
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Error getting GPT response: {str(e)}")
            raise Exception(f"GPT-4 evaluation failed: {str(e)}")

    def _parse_response(self, response: str) -> Dict[str, Any]:
        try:
            if "```json" in response:
                json_start = response.find("```json") + 7
                json_end = response.find("```", json_start)
                json_str = response[json_start:json_end].strip()
            elif "{" in response and "}" in response:
                json_start = response.find("{")
                json_end = response.rfind("}") + 1
                json_str = response[json_start:json_end]
            else:
                return {
                    "answer": response,
                    "rationale": "Response could not be parsed as JSON",
                    "confidence": 0.5
                }

            return json.loads(json_str)
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing JSON response: {str(e)}")
            return {
                "answer": response,
                "rationale": "Response parsing failed",
                "confidence": 0.3
            }
        except Exception as e:
            logger.error(f"Error parsing response: {str(e)}")
            return {
                "answer": response,
                "rationale": "Response processing error",
                "confidence": 0.3
            }
