"""
Workflow Code File

This file contains the workflow code that will be executed when testing workflows.
Copy and paste your generated workflow code here before clicking "Test Workflow".

Instructions:
1. Generate workflow code using the "Create Workflow" section
2. Copy the generated code from the UI
3. Paste it here, replacing this placeholder
4. Click "Test Workflow" to execute the code from this file

The workflow code should be a complete Python script that can be executed independently.
"""

# Paste your workflow code below this line

import asyncio
import aiohttp
import json
import logging
import re
from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta

# Configuration - replace with your actual values
LIGHTON_API_KEY = "your_api_key_here"
LIGHTON_BASE_URL = "https://api.lighton.ai"

logger = logging.getLogger(__name__)

class ParadigmClient:
    def __init__(self, api_key: str, base_url: str):
        self.base_url = base_url
        self.api_key = api_key
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
    
    async def document_search(self, query: str, **kwargs) -> Dict[str, Any]:
        endpoint = f"{self.base_url}/api/v2/chat/document-search"
        payload = {"query": query, **kwargs}
        
        async with aiohttp.ClientSession() as session:
            async with session.post(endpoint, json=payload, headers=self.headers) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    raise Exception(f"API error {response.status}: {await response.text()}")
    
    async def analyze_documents_with_polling(self, query: str, document_ids: List[int], **kwargs) -> str:
        # Start analysis
        endpoint = f"{self.base_url}/api/v2/chat/document-analysis"
        payload = {"query": query, "document_ids": document_ids, **kwargs}
        
        async with aiohttp.ClientSession() as session:
            async with session.post(endpoint, json=payload, headers=self.headers) as response:
                if response.status == 200:
                    result = await response.json()
                    chat_response_id = result.get("chat_response_id")
                else:
                    raise Exception(f"Analysis API error {response.status}: {await response.text()}")
        
        # Poll for results
        max_wait = 300  # 5 minutes
        poll_interval = 5
        elapsed = 0
        
        while elapsed < max_wait:
            endpoint = f"{self.base_url}/api/v2/chat/document-analysis/{chat_response_id}"
            async with aiohttp.ClientSession() as session:
                async with session.get(endpoint, headers=self.headers) as response:
                    if response.status == 200:
                        result = await response.json()
                        status = result.get("status", "")
                        if status.lower() in ["completed", "complete", "finished", "success"]:
                            analysis_result = result.get("result") or result.get("detailed_analysis") or "Analysis completed"
                            return analysis_result
                        elif status.lower() in ["failed", "error"]:
                            raise Exception(f"Analysis failed: {status}")
                    elif response.status == 404:
                        # Analysis not ready yet, continue polling
                        pass
                    else:
                        raise Exception(f"Polling API error {response.status}: {await response.text()}")
                    
                    await asyncio.sleep(poll_interval)
                    elapsed += poll_interval
        
        raise Exception("Analysis timed out")
    
    async def chat_completion(self, prompt: str, model: str = "alfred-4.2") -> str:
        endpoint = f"{self.base_url}/api/v2/chat/completions"
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ]
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(endpoint, json=payload, headers=self.headers) as response:
                if response.status == 200:
                    result = await response.json()
                    return result["choices"][0]["message"]["content"]
                else:
                    raise Exception(f"Paradigm chat completion API error {response.status}: {await response.text()}")
    
    async def analyze_image(self, query: str, document_ids: List[str], model: str = None, private: bool = False) -> str:
        endpoint = f"{self.base_url}/api/v2/chat/image-analysis"
        payload = {
            "query": query,
            "document_ids": document_ids
        }
        if model:
            payload["model"] = model
        if private is not None:
            payload["private"] = private
        
        async with aiohttp.ClientSession() as session:
            async with session.post(endpoint, json=payload, headers=self.headers) as response:
                if response.status == 200:
                    result = await response.json()
                    return result.get("answer", "No analysis result provided")
                else:
                    raise Exception(f"Image analysis API error {response.status}: {await response.text()}")

# Initialize clients
paradigm_client = ParadigmClient(LIGHTON_API_KEY, LIGHTON_BASE_URL)

async def execute_workflow(user_input: str) -> str:
    # Get attached files
    attached_file_ids = globals().get('attached_file_ids', [])
    
    if not attached_file_ids:
        return "ERROR: No documents uploaded. Please upload a payment request followed by invoice documents."
    
    if len(attached_file_ids) < 2:
        return "ERROR: At least 2 documents required - one payment request and at least one invoice."
    
    # Sort by document ID to determine upload order (smallest ID = first uploaded = payment request)
    sorted_file_ids = sorted(attached_file_ids)
    payment_request_id = sorted_file_ids[0]
    invoice_ids = sorted_file_ids[1:]
    
    try:
        # Step 1: Extract total payment amount from payment request
        # Use more specific and contextual queries
        payment_queries = [
            "Extract the total payment amount requested in this payment request. Look for words like 'total', 'amount', 'sum', 'payment', or currency symbols. Include any subtotals that need to be added together to get the final total amount to be paid.",
            "What is the total monetary value or amount being requested for payment in this document? Include all amounts that should be paid, whether shown as individual items or as a grand total.",
            "Find the payment amount, total amount, or sum to be paid in this payment request document. Look for numerical values with currency symbols or payment-related context."
        ]
        
        payment_content = ""
        payment_amount = 0
        payment_currency = ""
        
        # Try multiple search approaches for payment request
        for query in payment_queries:
            payment_search_result = await paradigm_client.document_search(
                query,
                file_ids=[payment_request_id]
            )
            
            content = payment_search_result.get("answer", "")
            if content and content.strip() and "no" not in content.lower() and "not found" not in content.lower():
                payment_content = content
                break
        
        # Try vision search as fallback if no content found
        if not payment_content or payment_content.strip() == "":
            visual_payment_result = await paradigm_client.document_search(
                "Extract the total payment amount from this payment request document. Look for numerical values with currency symbols.",
                file_ids=[payment_request_id],
                tool="VisionDocumentSearch"
            )
            payment_content = visual_payment_result.get("answer", "")
        
        # Extract payment amount using more detailed structured prompt
        payment_extraction_prompt = f"""Extract payment information from the following content and return valid JSON only.

Look for:
- Total amounts, grand totals, payment amounts
- Currency symbols (€, $, £, etc.) 
- Numbers with decimal points
- Words like "total", "amount", "sum", "payment"
- Multiple amounts that need to be summed

JSON SCHEMA:
{{
  "total_amount": "number or null - the final total amount to be paid",
  "currency": "string or null - currency symbol or code",
  "individual_amounts": "array of numbers found in the document",
  "found": "boolean - true if any amount was found"
}}

CONTENT: {payment_content}

If multiple amounts are present, determine if they are subtotals that should be summed to get the total payment amount. Return the final total amount in total_amount field.

JSON:"""

        payment_json_result = await paradigm_client.chat_completion(payment_extraction_prompt)
        
        payment_data = {"total_amount": None, "currency": None, "found": False, "individual_amounts": []}
        try:
            payment_data = json.loads(payment_json_result)
            if payment_data.get("individual_amounts") and not payment_data.get("total_amount"):
                # Sum individual amounts if total not found
                payment_data["total_amount"] = sum(payment_data.get("individual_amounts", []))
        except json.JSONDecodeError:
            # Parse using regex as fallback
            numbers = re.findall(r'[\d,]+\.?\d*', payment_content)
            if numbers:
                try:
                    amounts = [float(n.replace(',', '')) for n in numbers if '.' in n or len(n) > 2]
                    if amounts:
                        payment_data = {
                            "total_amount": max(amounts),  # Take the largest as likely total
                            "currency": re.search(r'[€$£¥]', payment_content),
                            "found": True,
                            "individual_amounts": amounts
                        }
                except ValueError:
                    pass
        
        payment_amount = payment_data.get("total_amount", 0) or 0
        payment_currency = payment_data.get("currency", "") or ""
        
        # Step 2: Extract total amounts from each invoice
        invoice_details = []
        total_invoice_amount = 0
        
        # Enhanced invoice queries
        invoice_queries = [
            "What is the total amount due on this invoice? Look for the final amount to be paid, grand total, or invoice total. Handle Arabic text if present.",
            "Extract the invoice total amount - the final sum that should be paid, not individual line items or subtotals. Include currency information.",
            "Find the total payment amount on this invoice document. Look for numerical values representing the amount owed."
        ]
        
        for invoice_id in invoice_ids:
            invoice_content = ""
            
            # Try multiple approaches for each invoice
            for query in invoice_queries:
                invoice_search_result = await paradigm_client.document_search(
                    query,
                    file_ids=[invoice_id]
                )
                
                content = invoice_search_result.get("answer", "")
                if content and content.strip() and "no" not in content.lower() and "not found" not in content.lower():
                    invoice_content = content
                    break
            
            # Try vision search as fallback
            if not invoice_content or invoice_content.strip() == "":
                visual_invoice_result = await paradigm_client.document_search(
                    "Extract the total amount from this invoice. Look for numerical values with currency symbols representing the amount to be paid.",
                    file_ids=[invoice_id],
                    tool="VisionDocumentSearch"
                )
                invoice_content = visual_invoice_result.get("answer", "")
            
            # Extract invoice details using enhanced prompt
            invoice_extraction_prompt = f"""Extract invoice information from the following content and return valid JSON only.

Look for:
- Invoice total, amount due, grand total
- Invoice number or identifier
- Invoice date
- Currency information
- Handle Arabic text appropriately

JSON SCHEMA:
{{
  "total_amount": "number or null - the total amount on this invoice",
  "currency": "string or null - currency symbol or code", 
  "invoice_date": "string or null - invoice date in YYYY-MM-DD format if found",
  "invoice_number": "string or null - invoice number or identifier",
  "found": "boolean - true if amount was found"
}}

CONTENT: {invoice_content}

JSON:"""

            invoice_json_result = await paradigm_client.chat_completion(invoice_extraction_prompt)
            
            invoice_data = {"total_amount": None, "currency": None, "invoice_date": None, "invoice_number": None, "found": False}
            try:
                invoice_data = json.loads(invoice_json_result)
            except json.JSONDecodeError:
                # Parse using regex as fallback
                numbers = re.findall(r'[\d,]+\.?\d*', invoice_content)
                if numbers:
                    try:
                        amounts = [float(n.replace(',', '')) for n in numbers if '.' in n or len(n) > 2]
                        if amounts:
                            invoice_data = {
                                "total_amount": max(amounts),  # Take the largest as likely total
                                "currency": re.search(r'[€$£¥]', invoice_content),
                                "found": True,
                                "invoice_number": f"Invoice {invoice_id}",
                                "invoice_date": None
                            }
                    except ValueError:
                        pass
            
            # Check if invoice is more than 90 days old
            is_old_invoice = False
            days_old = None
            if invoice_data.get("invoice_date"):
                try:
                    invoice_date = datetime.strptime(invoice_data["invoice_date"], "%Y-%m-%d")
                    current_date = datetime.now()
                    days_old = (current_date - invoice_date).days
                    is_old_invoice = days_old > 90
                except ValueError:
                    is_old_invoice = False
                    days_old = None
            
            invoice_amount = invoice_data.get("total_amount", 0) or 0
            if invoice_amount:
                total_invoice_amount += invoice_amount
            
            invoice_details.append({
                "document_id": invoice_id,
                "invoice_number": invoice_data.get("invoice_number", f"Invoice {invoice_id}"),
                "amount": invoice_amount,
                "currency": invoice_data.get("currency", ""),
                "date": invoice_data.get("invoice_date", "Not found"),
                "is_old": is_old_invoice,
                "days_old": days_old,
                "found": invoice_data.get("found", False)
            })
        
        # Step 3: Compare amounts (rounded to nearest unit)
        payment_amount_rounded = round(payment_amount)
        total_invoice_amount_rounded = round(total_invoice_amount)
        
        validation_result = "PASS" if payment_amount_rounded == total_invoice_amount_rounded else "FAIL"
        amount_difference = payment_amount_rounded - total_invoice_amount_rounded
        
        # Generate detailed report
        report = f"""PAYMENT REQUEST VALIDATION REPORT
{'='*50}

PAYMENT REQUEST DETAILS:
Document ID: {payment_request_id}
Total Amount: {payment_amount:.2f} {payment_currency}
Rounded Amount: {payment_amount_rounded}
Extraction Status: {"Found" if payment_amount > 0 else "Not Found - Check document content"}

INVOICE DETAILS:
"""
        
        old_invoices = []
        for invoice in invoice_details:
            status_indicator = "⚠️ OLD (>90 days)" if invoice["is_old"] else "✓"
            extraction_status = "Found" if invoice["found"] else "Not Found"
            report += f"""
Invoice: {invoice['invoice_number']} (ID: {invoice['document_id']})
Amount: {invoice['amount']:.2f} {invoice['currency']}
Date: {invoice['date']}
Status: {status_indicator}
Extraction: {extraction_status}"""
            if invoice["days_old"] is not None:
                report += f" ({invoice['days_old']} days old)"
            report += "\n"
            
            if invoice["is_old"]:
                old_invoices.append(invoice)
        
        report += f"""
SUMMARY:
Total Invoice Amount: {total_invoice_amount:.2f}
Total Invoice Amount (Rounded): {total_invoice_amount_rounded}
Payment Request Amount (Rounded): {payment_amount_rounded}
Difference: {amount_difference}

VALIDATION RESULT: {validation_result}
"""
        
        if validation_result == "FAIL":
            report += f"\n❌ MISMATCH DETECTED: Payment request amount ({payment_amount_rounded}) does not match total invoice amount ({total_invoice_amount_rounded})"
        else:
            report += f"\n✅ AMOUNTS MATCH: Payment request and invoices are aligned"
        
        if old_invoices:
            report += f"\n\n⚠️ OLD INVOICES DETECTED ({len(old_invoices)} invoices > 90 days old):"
            for old_invoice in old_invoices:
                report += f"\n- {old_invoice['invoice_number']}: {old_invoice['days_old']} days old"
            report += "\nNote: Old invoices are highlighted but do not cause validation failure."
        
        # Add debugging information if amounts weren't found
        if payment_amount == 0 or total_invoice_amount == 0:
            report += f"\n\n🔧 DEBUGGING INFO:"
            if payment_amount == 0:
                report += f"\nPayment request content preview: {payment_content[:200]}..."
            if total_invoice_amount == 0:
                report += f"\nConsider using VisionDocumentSearch tool or checking document parsing quality."
        
        return report
        
    except Exception as e:
        return f"ERROR: Failed to process payment validation: {str(e)}"

