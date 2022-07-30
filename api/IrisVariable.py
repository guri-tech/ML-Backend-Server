from pydantic import BaseModel

class IrisVariable(BaseModel):
    petal_length: float = 1.
    petal_width: float = 1.
    fetal_length: float = 1.
    fetal_width: float = 1.
