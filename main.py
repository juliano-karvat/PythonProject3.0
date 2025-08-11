import os
import json
from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

app = FastAPI(
    title="Function Calling - Livro ou Autor",
    description="Detecta se o texto é sobre um livro ou um autor e retorna o cartão correto.",
    version="1.0.0"
)

# ---- MODELOS ----
class ParagraphInput(BaseModel):
    text: str

class BookCard(BaseModel):
    title: str
    author: str
    year: int
    genre: str

class AuthorCard(BaseModel):
    name: str
    birth_year: int
    nationality: str
    notable_works: list[str]

# ---- FUNÇÕES DE CARTÃO ----
def make_book_card(title: str, author: str, year: int, genre: str):
    return BookCard(title=title, author=author, year=year, genre=genre)

def make_author_card(name: str, birth_year: int, nationality: str, notable_works: list[str]):
    return AuthorCard(name=name, birth_year=birth_year, nationality=nationality, notable_works=notable_works)

@app.post("/classify")
def classify_paragraph(data: ParagraphInput):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Você é um assistente que extrai dados estruturados sobre livros ou autores a partir de um texto."},
            {"role": "user", "content": data.text}
        ],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "make_book_card",
                    "description": "Cria um cartão JSON com informações sobre um livro",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "title": {"type": "string"},
                            "author": {"type": "string"},
                            "year": {"type": "integer"},
                            "genre": {"type": "string"}
                        },
                        "required": ["title", "author", "year", "genre"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "make_author_card",
                    "description": "Cria um cartão JSON com informações sobre um autor",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "birth_year": {"type": "integer"},
                            "nationality": {"type": "string"},
                            "notable_works": {"type": "array", "items": {"type": "string"}}
                        },
                        "required": ["name", "birth_year", "nationality", "notable_works"]
                    }
                }
            }
        ],
        tool_choice="auto"
    )

    choice = response.choices[0]
    if choice.finish_reason == "tool_calls":
        for tool_call in choice.message.tool_calls:
            func_name = tool_call.function.name
            args = json.loads(tool_call.function.arguments)

            if func_name == "make_book_card":
                return make_book_card(**args)
            elif func_name == "make_author_card":
                return make_author_card(**args)

    return {"error": "Não foi possível classificar o texto."}