"""rabbithole.planner module"""

import json

import openai


def generate_plan(summaries: dict[str, str], keywords: dict[str, list[str]]) -> dict:
    """
    Generate a plan from a list of summaries and keywords.
    :param summaries: Summaries for each document.
    :param keywords: List of keywords for each document.
    :return: Generated plan.
    """

    def format_document(name: str, summary: str, keywords: list[str]) -> str:
        """Format a document for the plan."""

        # Create a string of comma-separated keywords
        keywords_str = ', '.join(keywords)

        # Construct the formatted document
        document = f"{name}\n{'=' * len(name)}\n{summary}\nKeywords: {keywords_str}\n\n"

        return document

    prompt = f"""
    I have the following documents with summaries and keywords:
    {''.join([format_document(name, summary, keywords) for name, summary, keywords in zip(summaries.keys(), summaries.values(), keywords.values())])}
    \n\n
    The plan should be formatted as follows:
    
    {{
        "plan": [
            {{
                "Document 1": {{
                    "Background Concepts": ["Concept 1", "Concept 2", ...],
                    "Key concepts": ["Concept 1", "Concept 2", ...],
                    "Further reading": ["Concept 1", "Concept 2", ...]
                }}
            }}
            ...
        ]
    }}
    
    Only provide the JSON response and do not provide any other information other than it.
    Make sure the JSON is formatted correctly and without any errors.
    """

    print("Making a request to OpenAI's API to generate a plan...")
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system",
                 "content": "You are an experienced professor helping a student plan their studies. You know everything about all subjects and need to answer factually and logically."},
                {"role": "user", "content": prompt}
            ],
        )
        print(response)
    except Exception as e:
        print(e)
        return {
            "plan": "Error generating a plan.\nPlease try again."
        }

    # Get the JSON response
    response = response.choices[0].message.content
    first_bracket = response.find('{')
    last_bracket = response.rfind('}')

    # Convert the JSON response to a dictionary
    return json.loads(response[first_bracket:last_bracket + 1])
