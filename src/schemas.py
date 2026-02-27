from pydantic import BaseModel

class LoanInput(BaseModel):
    applicant_income: float
    coapplicant_income: float
    loan_amount: float
    loan_term: float
    credit_history: float