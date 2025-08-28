# app/schema.py

from pydantic import BaseModel

class TitanicPassenger(BaseModel):
    Pclass: int
    Age: float
    SibSp: int
    Parch: int
    Fare: float
    SexIndexed: float
    EmbarkedIndexed: float
