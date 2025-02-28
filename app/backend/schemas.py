from pydantic import BaseModel
from typing import Optional, Literal
from datetime import datetime

class Questao(BaseModel):
    id : int
    type : str
    year : int
    discipline : str
    context : str
    context_image : Optional[str] = None
    alternatives_introduction : str
    correct_alternative : str
    A : str
    A_file : Optional[str] = None
    B : str
    B_file : Optional[str] = None
    C : str
    C_file : Optional[str] = None
    D : str
    D_file : Optional[str] = None
    E : str
    E_file : Optional[str] = None

class Chat(BaseModel):
    id : str
    name : str
    question : str
    model : str
    last_update : datetime = datetime.now()
    messages : list[dict]

class Model(BaseModel):
    name : str
    image : str
    