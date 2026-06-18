import time

from google.genai import errors, types

INSTRUCTIONS = """
Your task is to answer questions from the course participants
based on the provided context.

Use the context to find relevant information and provide accurate
answers. If the answer is not found in the context,
respond with "I don't know."
"""

USER_PROMPT_TEMPLATE = """
QUESTION: 
{question}

CONTEXT:
{context}
"""

DEFAULT_MODELS = [
    "gemini-2.5-flash-lite",
    "gemini-3.1-flash-lite",
    "gemma-4-26b-a4b-it",
]


class RAGBase:

    def __init__(
        self,
        index,
        genai_client,
        instructions=INSTRUCTIONS,
        prompt_template=USER_PROMPT_TEMPLATE,
        course="llm-zoomcamp",
        model="gemini-2.5-flash-lite",
        fallback_models=None,
    ):
        self.index = index
        self.genai_client = genai_client
        self.instructions = instructions
        self.course = course
        self.prompt_template = prompt_template
        self.model = model
        self.fallback_models = fallback_models or DEFAULT_MODELS

    def search(self, query, num_results=5):
        boost_dict = {"question": 3.0, "section": 0.5}
        filter_dict = {"course": self.course}

        return self.index.search(
            query,
            num_results=num_results,
            boost_dict=boost_dict,
            filter_dict=filter_dict,
        )

    def build_context(self, search_results):
        lines = []

        for doc in search_results:
            lines.append(doc["section"])
            lines.append("Q: " + doc["question"])
            lines.append("A: " + doc["answer"])
            lines.append("")

        return "\n".join(lines).strip()

    def build_prompt(self, query, search_results):
        context = self.build_context(search_results)
        return self.prompt_template.format(
            question=query, context=context
        )

    def llm(self, prompt):
        config = types.GenerateContentConfig(
            system_instruction=self.instructions.strip(),
        )

        models = [self.model] + [
            m for m in self.fallback_models if m != self.model
        ]

        last_error = None
        for model in models:
            for attempt in range(3):
                try:
                    response = self.genai_client.models.generate_content(
                        model=model,
                        contents=prompt,
                        config=config,
                    )
                    return response.text
                except errors.ClientError as e:
                    last_error = e
                    if e.code in (429, 503):
                        time.sleep(2 ** attempt)
                        continue
                    raise
                except errors.ServerError as e:
                    last_error = e
                    time.sleep(2 ** attempt)
                    continue

        raise last_error

    def rag(self, query):
        search_results = self.search(query)
        prompt = self.build_prompt(query, search_results)
        return self.llm(prompt)
