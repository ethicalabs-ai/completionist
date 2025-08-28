from pydantic import BaseModel, Field


class DefaultSchema(BaseModel):
    """A default Pydantic schema with a prompt and a completion."""

    prompt: str = Field(..., description="The generated user prompt or query.")
    completion: str = Field(..., description="The generated model completion.")


class SchemaWithReasoning(BaseModel):
    """A Pydantic schema that includes a reasoning or 'thought' process field."""

    prompt: str = Field(..., description="The generated user prompt or query.")
    completion: str = Field(..., description="The generated model completion.")
    reasoning: str = Field(..., description="The model's reasoning or thought process.")
