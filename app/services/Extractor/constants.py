"""
Constants and Configuration
SWIFT MT Message Types, Field Mappings, and Supporting Document Types
"""

# ============================================================================
# SWIFT MT MESSAGE TYPES
# ============================================================================

SWIFT_MESSAGE_TYPES = {
    '700': {'name': 'Issue of a Documentary Credit', 'category': 'Documentary Credits'},
    '701': {'name': 'Issue of a Documentary Credit', 'category': 'Documentary Credits'},
    '705': {'name': 'Pre-Advice of a Documentary Credit', 'category': 'Documentary Credits'},
    '707': {'name': 'Amendment to a Documentary Credit', 'category': 'Documentary Credits'},
    '710': {'name': 'Advice of Third Bank LC', 'category': 'Documentary Credits'},
    '720': {'name': 'Transfer of a Documentary Credit', 'category': 'Documentary Credits'},
    '730': {'name': 'Acknowledgement', 'category': 'Documentary Credits'},
    '740': {'name': 'Authorization to Reimburse', 'category': 'Documentary Credits'},
    '747': {'name': 'Amendment to Authorization to Reimburse', 'category': 'Documentary Credits'},
    '750': {'name': 'Advice of Discrepancy', 'category': 'Documentary Credits'},
    '760': {'name': 'Issue of a Guarantee', 'category': 'Guarantees'},
    '767': {'name': 'Amendment to a Guarantee', 'category': 'Guarantees'},
    '780': {'name': 'Claim under a Guarantee', 'category': 'Guarantees'},
    '790': {'name': 'Advice of Charges/Interest', 'category': 'Documentary Credits'},
}


# ============================================================================
# COMPREHENSIVE FIELD MAPPINGS
# ============================================================================

COMPREHENSIVE_FIELD_MAPPINGS = {
    ':20:': 'Transaction Reference Number',
    ':21:': 'Related Reference',
    ':23:': 'Issuing Bank\'s Reference',
    ':26E:': 'Number of Amendment',
    ':27:': 'Sequence of Total',
    ':30:': 'Date of Amendment',
    ':31C:': 'Date of Issue',
    ':31D:': 'Date and Place of Expiry',
    ':32A:': 'Value Date/Currency/Amount',
    ':32B:': 'Currency/Amount',
    ':39A:': 'Percentage Credit Amount Tolerance',
    ':40A:': 'Form of Documentary Credit',
    ':40E:': 'Applicable Rules',
    ':41A:': 'Available With...By...',
    ':42P:': 'Deferred Payment Details',
    ':43P:': 'Partial Shipments',
    ':43T:': 'Transhipment',
    ':44C:': 'Latest Date of Shipment',
    ':44E:': 'Port of Loading',
    ':44F:': 'Port of Discharge',
    ':45A:': 'Description of Goods',
    ':45B:': 'Description of Goods (Amendment)',
    ':46A:': 'Documents Required',
    ':46B:': 'Documents Required (Amendment)',
    ':47A:': 'Additional Conditions',
    ':47B:': 'Additional Conditions (Amendment)',
    ':48:': 'Period for Presentation',
    ':49:': 'Confirmation Instructions',
    ':50:': 'Applicant',
    ':51D:': 'Applicant Bank',
    ':52A:': 'Issuing Bank',
    ':52D:': 'Issuing Bank',
    ':53A:': 'Reimbursing Bank',
    ':57A:': 'Advise Through Bank',
    ':59:': 'Beneficiary',
    ':71D:': 'Details of Charges',
    ':72Z:': 'Sender to Receiver Information',
    ':78:': 'Instructions to Paying/Accepting/Negotiating Bank',
    ':22A:': 'Purpose of Message',
}


# ============================================================================
# SUPPORTING DOCUMENT TYPES
# ============================================================================

SUPPORTING_DOCUMENT_TYPES = {

    # =========================
    # FINANCIAL / COMMERCIAL
    # =========================
    "COMMERCIAL_INVOICE": [
        "COMMERCIAL INVOICE",
        "FINAL INVOICE",
        "INVOICE NO",
        "TAX INVOICE",
        "PROFORMA INVOICE"
    ],

    "PROFORMA_INVOICE": [
        "PROFORMA INVOICE"
    ],

    "CREDIT_NOTE": [
        "CREDIT NOTE"
    ],

    "DEBIT_NOTE": [
        "DEBIT NOTE"
    ],

    "STATEMENT_OF_ACCOUNT": [
        "STATEMENT OF ACCOUNT"
    ],

    # =========================
    # TRANSPORT DOCUMENTS
    # =========================
    "BILL_OF_LADING": [
        "BILL OF LADING",
        "B/L",
        "OCEAN BILL OF LADING",
        "SHIPPED ON BOARD BILL OF LADING",
        "CLEAN ON BOARD"
    ],

    "SEA_WAYBILL": [
        "SEA WAYBILL",
        "SEAWAY BILL"
    ],

    "AIR_WAYBILL": [
        "AIR WAYBILL",
        "AWB",
        "MASTER AIR WAYBILL",
        "HOUSE AIR WAYBILL"
    ],

    "ROAD_TRANSPORT_DOCUMENT": [
        "CMR",
        "ROAD CONSIGNMENT NOTE"
    ],

    "RAIL_TRANSPORT_DOCUMENT": [
        "RAILWAY RECEIPT",
        "RAIL CONSIGNMENT NOTE"
    ],

    "MULTIMODAL_TRANSPORT_DOCUMENT": [
        "MULTIMODAL TRANSPORT DOCUMENT",
        "COMBINED TRANSPORT DOCUMENT"
    ],

    "CHARTER_PARTY_BILL": [
        "CHARTER PARTY BILL OF LADING"
    ],

    # =========================
    # PACKING / QUANTITY
    # =========================
    "PACKING_LIST": [
        "PACKING LIST",
        "PACKING DETAILS"
    ],

    "WEIGHT_CERTIFICATE": [
        "WEIGHT CERTIFICATE",
        "CERTIFICATE OF WEIGHT"
    ],

    "MEASUREMENT_CERTIFICATE": [
        "MEASUREMENT CERTIFICATE",
        "ULLAGE REPORT",
        "DRAFT SURVEY"
    ],

    "QUANTITY_CERTIFICATE": [
        "QUANTITY CERTIFICATE"
    ],

    # =========================
    # INSURANCE
    # =========================
    "INSURANCE_POLICY": [
        "INSURANCE POLICY"
    ],

    "INSURANCE_CERTIFICATE": [
        "INSURANCE CERTIFICATE"
    ],

    "COVER_NOTE": [
        "COVER NOTE"
    ],

    # =========================
    # ORIGIN / TRADE COMPLIANCE
    # =========================
    "CERTIFICATE_OF_ORIGIN": [
        "CERTIFICATE OF ORIGIN",
        "COO",
        "FORM A",
        "FORM E"
    ],

    "EUR1_CERTIFICATE": [
        "EUR.1",
        "EUR1 CERTIFICATE"
    ],

    "ATR_CERTIFICATE": [
        "ATR CERTIFICATE"
    ],

    "GSP_CERTIFICATE": [
        "GSP CERTIFICATE"
    ],

    # =========================
    # INSPECTION / QUALITY
    # =========================
    "INSPECTION_CERTIFICATE": [
        "INSPECTION CERTIFICATE",
        "CERTIFICATE OF INSPECTION"
    ],

    "QUALITY_CERTIFICATE": [
        "CERTIFICATE OF QUALITY",
        "CERTIFICATE OF ANALYSIS",
        "ANALYSIS CERTIFICATE"
    ],

    "SURVEYOR_REPORT": [
        "SURVEYOR REPORT",
        "SURVEY REPORT"
    ],

    "PRE_SHIPMENT_INSPECTION": [
        "PRE-SHIPMENT INSPECTION",
        "PSI CERTIFICATE"
    ],

    # =========================
    # AGRICULTURE / FOOD / HEALTH
    # =========================
    "PHYTOSANITARY_CERTIFICATE": [
        "PHYTOSANITARY CERTIFICATE"
    ],

    "VETERINARY_CERTIFICATE": [
        "VETERINARY CERTIFICATE"
    ],

    "HEALTH_CERTIFICATE": [
        "HEALTH CERTIFICATE"
    ],

    "FUMIGATION_CERTIFICATE": [
        "FUMIGATION CERTIFICATE"
    ],

    # =========================
    # BENEFICIARY / APPLICANT DECLARATIONS
    # =========================
    "BENEFICIARY_CERTIFICATE": [
        "BENEFICIARY CERTIFICATE"
    ],

    "APPLICANT_CERTIFICATE": [
        "APPLICANT CERTIFICATE"
    ],

    "CERTIFICATE_OF_COMPLIANCE": [
        "CERTIFICATE OF COMPLIANCE"
    ],

    "NON_MANIPULATION_CERTIFICATE": [
        "NON MANIPULATION CERTIFICATE"
    ],

    "NON_BLACKLIST_CERTIFICATE": [
        "NON BLACKLIST CERTIFICATE"
    ],

    # =========================
    # CUSTOMS / REGULATORY
    # =========================
    "CUSTOMS_DECLARATION": [
        "CUSTOMS DECLARATION",
        "EXPORT DECLARATION",
        "IMPORT DECLARATION"
    ],

    "EXPORT_LICENSE": [
        "EXPORT LICENSE"
    ],

    "IMPORT_LICENSE": [
        "IMPORT LICENSE"
    ],

    # =========================
    # PAYMENT / BANKING
    # =========================
    "BENEFICIARY_STATEMENT": [
        "BENEFICIARY STATEMENT"
    ],

    "DRAFT": [
        "DRAFT",
        "BILL OF EXCHANGE"
    ],

    "REMITTANCE_ADVICE": [
        "REMITTANCE ADVICE"
    ],

    # =========================
    # MISC / FALLBACK
    # =========================
    "LETTER_OF_INDEMNITY": [
        "LETTER OF INDEMNITY",
        "LOI"
    ],

    "DELIVERY_ORDER": [
        "DELIVERY ORDER"
    ],

    "WAREHOUSE_RECEIPT": [
        "WAREHOUSE RECEIPT"
    ],

    "UNKNOWN_SUPPORTING": []
}