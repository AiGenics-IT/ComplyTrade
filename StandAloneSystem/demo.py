"""
Demo script to process the provided LC documents
"""

import sys
import json
from pathlib import Path

# Add current directory to path
sys.path.insert(0, '/home/claude')

from lc_extractor import LCExtractor, LCConsolidator

def demo_with_text_data():
    """Demo using the text content from uploaded documents"""
    
    print("=" * 80)
    print("LC PROCESSING DEMO - Using Provided Documents")
    print("=" * 80)
    print()
    
    # Original LC text (MT700)
    lc_text = """
Messages
Message 1
Message Identifier
Message Preparation 
Application:
Alliance Message Management
Unique Message Identifier: I EBILSGSGXXX 700 0239ILU012702 (suffix 2109024064249)
Message Header
Status: Message Modified
Deletable
Format: Swift Sub-Format: Input
Identifier: fin.700 Expansion: Issue of a Documentary Credit
Application FIN Nature: Financial
Sender: UNILPKKA741 LT: A
Receiver: EBILSGSGXXX LT: X
Transaction Reference: 0239ILU012702
Priority: Normal
Monitoring: None
MUR: SAL/LC
Amount: 52,069,920. Currency: USD Value Date:
Sender / Receiver
Sender Institution: UNILPKKA741 Expansion: UNITED BANK LIMITED
(JINNAH AVENUE BRANCH)
ISLAMABAD
ISLAMABAD
PK
PAKISTAN
Receiver Institution: EBILSGSGXXX Expansion: EMIRATES NBD BANK PJSC (ENBD)
SINGAPORE 049315
SINGAPORE
SG
SINGAPORE
Message Text
Block 4
F27: Sequence of Total
 Number: 1/ 
 Total: 1 
F40A: Form of Documentary Credit
 IRREVOCABLE 
F20: Documentary Credit Number
 0239ILU012702 
F31C: Date of Issue
 210906 2021 Sep 06
F40E: Applicable Rules
 Applicable Rules: UCP LATEST VERSION 
F31D: Date and Place of Expiry
 Date: 211201 2021 Dec 01
 Place: SINGAPORE 
F51D: Applicant Bank - Party Identifier - Name and Address
 Name and Address:
 UNITED BANK LIMITED
 JINNAH AVENUE(0239)
 ISLAMABAD PAKISTAN
F50: Applicant
 PAKISTAN LNG LIMITED
 9TH FLOOR,PETROLEUM HOUSE,ATATURK
 AVENUE,G-5/2,ISLAMABAD-44000, 
 PAKISTAN.
F59: Beneficiary
 Name and Address:
 GUNVOR SINGAPORE PTE LTD
 12 MARINA BOULEVARD NO. 35-03,
 MARINA BAY FINANCIAL CENTER TOWER 
 3, SINGAPORE. 018982.
F32B: Currency Code, Amount
 Currency: USD US DOLLAR
 Amount: 52069920,00
F41D: Available With ... By ... - Name and Address - Code
 Name and Address:
 EMIRATES NBD BANK PJSC (ENBD)
 SINGAPORE
 EBILSGSGXXX
 Code: BY DEF PAYMENT 
F42P: Negotiation/Deferred Payment Details
 21 DAYS FROM THE DATE OF COMPLETION
 OF DISCHARGE AT DISCHARGE PORT AS
 PER FINAL DISCHARGE REPORT.
F43P: Partial Shipments
 NOT ALLOWED 
F43T: Transhipment
 NOT ALLOWED 
F44E: Port of Loading/Airport of Departure
 GATE 
F44F: Port of Discharge/Airport of Destination
 PORT QASIM, PAKISTAN 
F44C: Latest Date of Shipment
 210913 2021 Sep 13
F45A: Description of Goods and/or Services
 DESCRIPTION: LIQUEFIED NATURAL GAS (LNG)
 UNIT PRICE: USD 15.4970 PER MMBTU
 QUANTITY: 3,200,000 (-/+5%)
 DES PORT QASIM, PAKISTAN
 HS CODE 2711.1100
F46A: Documents Required
 1.SIGNED COMMERCIAL INVOICES IN ONE ORIGINALS & COPY SHOWING
 VALUE OF THE GOODS AS PER THE PRICE CLAUSE OF THE L/C MULTIPLIED
 BY THE QUANTITY RECEIVED AT DISCHARGE PORT IN MMBTU ALONG WITH
 ANY ADJUSTMENTS ON ACCOUNT OF PILOTAGE & TOWAGE FEE AND MONSOON
 CHARGES AS STATED IN THE PROVISIONAL INVOICE ISSUED BY PORT
 QASIM AUTHORITY IN ACCORDANCE WITH CLAUSE 18 OF THE CONFIRMATION
 NOTICE AND STATING WE CERTIFY THAT GOODS HEREIN INVOICED CONFORM
 WITH PROFORMA INVOICE AND ARE OF NETHERLANDS ORIGIN.
 .
 2.3/3 ORIGINAL + 3 NON-NEGOTIABLE BILL OF LADING DRAWN OR
 ENDORSED TO THE ORDER OF UNITED BANK LIMITED AND MARKED NOTIFY
 PAKISTAN LNG LIMITED.
 .
 3.ONE ORIGINAL + 3 COPIES OF CERTIFICATE OF ORIGIN ISSUED OR
 COUNTERSIGNED BY THE RELEVANT CHAMBER OF COMMERCE OR PORT
 AUTHORITIES OR TERMINAL OPERATORS (ISSUED IN THE NAME OF THE
 APPLICANT AS CONSIGNEE ACCEPTABLE) MENTIONING GOODS OF
 NETHERLANDS ORIGIN.
F47A: Additional Conditions
 1.PRICE CLAUSE:
 THE DELIVERED EX-SHIP CONTRACT PRICE FOR LNG CARGO SHALL BE
 CALCULATED AS FOLLOWS:
 CP = 15.4970 USD/MMBTU
 .
 2.ALL PARTIES TO THIS TRANSACTION ARE ADVISED THAT THE U.S AND
 OTHER GOVERNMENT AND/OR REGULATORY AUTHORITIES IMPOSE SPECIFIC
 SANCTIONS AGAINST CERTAIN COUNTRIES, ENTITIES AND INDIVIDUALS.
 .
 3.NAME OF THE DOCUMENTS EXCEPT B/L AND COMMERCIAL INVOICE
 DIFFERENT FROM L/C BUT SERVE THE SAME PURPOSE ARE ACCEPTABLE.
 .
 11.DOCUMENTS SHOWING PORT OF LOADING AS ''GATE'' IS
 ACCEPTABLE.
 .
 12.PORT OF DISCHARGE AS ''PORT QASIM PAKISTAN'' OR 
 ''PAKISTANPORT QASIM'' OR ''PORT QASIM'' OR ''PORT QASIM 
 KARACHI'' OR ''KARACHI, PAKISTAN'' ARE ACCEPTABLE.
"""
    
    # Amendment text (MT707)
    amendment_text = """
Messages
Message 1
Message Identifier
Message Preparation 
Application:
Alliance Message Management
Unique Message Identifier: I EBILSGSGXXX 707 0239ILU012702 (suffix 2109084075627)
Message Header
Status: Message Modified
Deletable
Format: Swift Sub-Format: Input
Identifier: fin.707 Expansion: Amendment to a Documentary Credit
Application FIN Nature: Financial
Sender: UNILPKKA741 LT: A
Receiver: EBILSGSGXXX LT: X
Transaction Reference: 0239ILU012702
Related Reference: ENSGLA21000518
Priority: Normal
Monitoring: None
MUR: QA/LC
Sender / Receiver
Sender Institution: UNILPKKA741 Expansion: UNITED BANK LIMITED
(JINNAH AVENUE BRANCH)
ISLAMABAD
ISLAMABAD
PK
PAKISTAN
Receiver Institution: EBILSGSGXXX Expansion: EMIRATES NBD BANK PJSC (ENBD)
SINGAPORE 049315
SINGAPORE
SG
SINGAPORE
Message Text
Block 4
F27: Sequence of Total
 Number: 1/ 
 Total: 1 
F20: Sender's Reference
 0239ILU012702 
F21: Receiver's Reference
 ENSGLA21000518 
F23: Issuing Bank's Reference
 0239ILU012702 
F52D: Issuing Bank - Party Identifier - Name and Address
 Name and Address:
 UNITED BANK LIMITED
 JINNAH AVENUE (0239)
 ISLAMABAD PAKISTAN
F31C: Date of Issue
 210906 2021 Sep 06
F26E: Number of Amendment
 01 
F30: Date of Amendment
 210908 2021 Sep 08
F22A: Purpose of Message
 ISSU 
F59: Beneficiary
 Name and Address:
 GUNVOR SINGAPORE PTE LTD
 12 MARINA BOULEVARD NO. 35-03,
 MARINA BAY FINANCIAL CENTER TOWER 
 3, SINGAPORE 018982
F46B: Documents Required
Line1: Line 1
 Code: /DELETE/ 
 Narrative: PLEASE READ WORDS IN FIELD 46A-1 ''CONFORM WITH 
Lines2to100: Lines 2-100
 Lines2to100: Lines 2-100: Narrative 
 Narrative: PROFORMA INVOICE AND'' AS DELETED. 
F47B: Additional Conditions
Line1: Line 1
 Code: /REPALL/ 
 Narrative: PLEASE READ FIELD 47A-11 AS ''DOCUMENTS 
Lines2to100: Lines 2-100
 Lines2to100: Lines 2-100: Narrative 
 Narrative: SHOWING PORT OF LOADING AS ''GATE'' 
 Lines2to100: Lines 2-100: Narrative 
 Narrative: OR ''GATE (ROTTERDAM)'' OR ''ROTTERDAM'' 
 Lines2to100: Lines 2-100: Narrative 
 Narrative: OR ''ONE SAFE PORT,ROTTERDAM'' OR 
 Lines2to100: Lines 2-100: Narrative 
 Narrative: ''GATE, ROTTERDAM'' ARE ACCEPTABLE. 
 Lines2to100: Lines 2-100: Code - Narrative 
 Code: /ADD/ 
 Narrative1: Narrative: PLEASE READ FIELD 47A-19 AS ''COMBINED 
 Lines2to100: Lines 2-100: Narrative 
 Narrative: OR SEPARATE DOCUMENTS PRESENTED ARE ACCEPTABLE. 
 Lines2to100: Lines 2-100: Code - Narrative 
 Code: /ADD/ 
 Narrative1: Narrative: PLEASE READ FIELD 47A-20 AS ''CERTIFICATE 
 Lines2to100: Lines 2-100: Narrative 
 Narrative: OF ORIGIN WITH ORIGIN AS ''THE NETHERLANDS'' 
 Lines2to100: Lines 2-100: Narrative 
 Narrative: IS ACCEPTABLE. 
 Lines2to100: Lines 2-100: Code - Narrative 
 Code: /ADD/ 
 Narrative1: Narrative: PLEASE READ FIELD 47A-21 AS ''CERTIFICATE 
 Lines2to100: Lines 2-100: Narrative 
 Narrative: OF ORIGIN ISSUED OR COUNTER SIGNED BY 
 Lines2to100: Lines 2-100: Narrative 
 Narrative: STEDER GROUP AGENCIES BV IS ACCEPTABLE. 
F72Z: Sender to Receiver Information
 ALL OTHER TERMS AND CONDITIONS OF 
 THIS LC SHALL REMAIN THE SAME.
"""
    
    # Initialize extractor and consolidator
    extractor = LCExtractor()
    consolidator = LCConsolidator()
    
    # Process original LC
    print("Processing Original LC...")
    print("-" * 80)
    lc_doc = extractor.extract_from_text(lc_text)
    
    print(f"Document Type: {lc_doc.document_type}")
    print(f"LC Number: {lc_doc.lc_number}")
    print(f"Issue Date: {lc_doc.issue_date}")
    print(f"Sender: {lc_doc.sender}")
    print(f"Receiver: {lc_doc.receiver}")
    print(f"Fields Extracted: {len(lc_doc.fields)}")
    print(f"Additional Conditions (F47A): {len(lc_doc.additional_conditions)} points")
    print(f"Documents Required (F46A): {len(lc_doc.documents_required)} points")
    print()
    
    print("Additional Conditions:")
    for condition in lc_doc.additional_conditions:
        print(f"  {condition['point_number']}. {condition['text'][:80]}...")
    print()
    
    consolidator.add_document(lc_doc)
    
    # Process amendment
    print("\nProcessing Amendment...")
    print("-" * 80)
    amendment_doc = extractor.extract_from_text(amendment_text)
    
    print(f"Document Type: {amendment_doc.document_type}")
    print(f"LC Number: {amendment_doc.lc_number}")
    print(f"Amendment Number: {amendment_doc.amendment_number}")
    print(f"Amendment Date: {amendment_doc.amendment_date}")
    print(f"Changes to Additional Conditions: {len(amendment_doc.additional_conditions)}")
    print(f"Changes to Documents Required: {len(amendment_doc.documents_required)}")
    print()
    
    print("Amendment Changes:")
    for change in amendment_doc.additional_conditions:
        print(f"  Operation: {change['operation']}")
        if 'point_number' in change:
            print(f"  Point: {change['point_number']}")
        print(f"  Details: {change['narrative'][:100]}...")
        print()
    
    consolidator.add_document(amendment_doc)
    
    # Consolidate
    print("\nConsolidating LC with Amendment...")
    print("-" * 80)
    consolidated = consolidator.consolidate(lc_doc.lc_number)
    
    print(f"LC Number: {consolidated['lc_number']}")
    print(f"Original Issue Date: {consolidated['original_issue_date']}")
    print(f"Amendments Applied: {consolidated['amendments_applied']}")
    print(f"Last Amendment Date: {consolidated['last_amendment_date']}")
    print(f"Final Additional Conditions: {len(consolidated['additional_conditions'])} points")
    print(f"Final Documents Required: {len(consolidated['documents_required'])} points")
    print()
    
    print("Final Additional Conditions (After Amendment):")
    for condition in consolidated['additional_conditions']:
        marker = ""
        if condition.get('added_by_amendment'):
            marker = " [ADDED]"
        elif condition.get('modified_by_amendment'):
            marker = " [MODIFIED]"
        
        print(f"  {condition['point_number']}. {condition['text'][:100]}...{marker}")
    print()
    
    # Save outputs
    output_dir = Path('/home/claude/demo_output')
    output_dir.mkdir(exist_ok=True)
    
    # Save original LC
    with open(output_dir / 'original_lc.json', 'w') as f:
        from dataclasses import asdict
        json.dump(asdict(lc_doc), f, indent=2, ensure_ascii=False)
    
    # Save amendment
    with open(output_dir / 'amendment_01.json', 'w') as f:
        json.dump(asdict(amendment_doc), f, indent=2, ensure_ascii=False)
    
    # Save consolidated
    with open(output_dir / 'consolidated_lc.json', 'w') as f:
        json.dump(consolidated, f, indent=2, ensure_ascii=False)
    
    print("=" * 80)
    print("DEMO COMPLETE")
    print("=" * 80)
    print(f"\nOutput files saved to: {output_dir}")
    print("  - original_lc.json")
    print("  - amendment_01.json")
    print("  - consolidated_lc.json")
    print()
    
    return consolidated


if __name__ == "__main__":
    demo_with_text_data()
