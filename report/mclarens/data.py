from pydantic import BaseModel


class MclarensInsuranceReportData(BaseModel):
    loss_date: str
    first_contact_date: str
    site_visit_date: str
    first_report_date: str
    estimate: str
    final_claim_paid: str
    date_final_settlement_agreed: str
    date_claim_closed: str
    date_of_first_payment: str
    cause_of_loss_level1: str
    cause_of_loss_level2: str
    instruction_date: str
    broker_name: str
    threshold_payments_completed_date: str
