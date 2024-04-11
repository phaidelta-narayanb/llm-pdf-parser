from ..base import Report
from .data import MclarensInsuranceReportData


class MclarensReport(Report):
    pass


class BaseMclarensReport(MclarensReport):
    report_type: str
    '''
    Report type (in heading).
    Example: Preliminary
    '''

    report_title: str
    '''
    Full report title.
    Example: Preliminary Loss Advice: Report 1
    '''

    report_date: str
    '''
    Date mentioned on the report.
    '''

    file_number: str
    '''
    Reference file number.
    Example: 000.111111.22
    '''


class BaseMclarensInsuranceReport(BaseMclarensReport):
    report_data: MclarensInsuranceReportData
