from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain_openai.chat_models import ChatOpenAI

knowledge_base = """
Germany
- capital: Berlin
- language: German
- population: 85 million
- currency: Euro

Belgium
- capital: Brussels
- languages: Dutch, French, German
- population: 12 million
- currency: Euro

France
- capital: Paris
- language: French
- population: 68 million
- currency: Euro

United Kingdom of Great Britain and Northern Ireland
- capital: London
- language: English
- population: 68 million
- currency: Pound
"""

system_message = f"""
Follow these steps to generate some fun facts for the user.

Step 1: Make sure the user asks about one of these countries. The user is only allowed to ask about these countries:
* Germany
* Belgium
* France
* United Kingdom

Step 2: For the selected country, pick detail information from the knowledge base below:

{knowledge_base}

Step 3: Generate three facts for the user. Based on the selected country and the given information, generate list of 3 facts.
Only reference facts from the included knowledge base.
Use the following format:
Fact 1: <fact 1>
---
Fact 2: <fact 2>
---
Fact 3: <fact 3>

IMPORTANT:
- the user is only allowed to ask about the listed countries, otherwise you should reply ""no country, no facts"
- it is dangerous to answer about anything else then the listed countries from the knowledge base.
"""


def llm_chain():
    human_template = "{question}"
    chat_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_message),
            ("human", human_template),
        ]
    )
    return (
        chat_prompt
        | ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
        | StrOutputParser()
    )
