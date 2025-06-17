import pandas as pd
import json
from typing import Dict, List, Optional, Tuple
from langchain_aws import ChatBedrock
import boto3
import os
from langchain.schema import HumanMessage, SystemMessage
from dotenv import load_dotenv
load_dotenv(".env")

class AirportExtractor:
    def __init__(self, aws_region: str = "us-east-1"):
        """
        Initialize the AirportExtractor with AWS Bedrock client
        
        Args:
            aws_region: AWS region for Bedrock service
        """
        self.aws_access_key = os.getenv("aws_access_key")
        self.aws_secret_access_key = os.getenv("aws_secret_access_key")
        self.client = boto3.client(
            "bedrock-runtime",
            aws_access_key_id=self.aws_access_key,
            aws_secret_access_key=self.aws_secret_access_key,
            region_name=aws_region
        )
        self.aws_region = aws_region
        self.llm = ChatBedrock(
            model_id="amazon.nova-pro-v1:0",
            region_name=aws_region,
            model_kwargs={
                "temperature": 0.5,
                "max_tokens": 1000
            },
            client=self.client
        )
    
    def load_airport_data(self, csv_file_path: str) -> pd.DataFrame:
        """
        Load airport data from CSV file
        
        Args:
            csv_file_path: Path to the CSV file
            
        Returns:
            DataFrame with airport data
        """
        try:
            df = pd.read_csv(csv_file_path)
            df.columns = df.columns.str.strip()
            for col in df.columns:
                if df[col].dtype == 'object':
                    df[col] = df[col].astype(str).str.strip()
            return df
        except Exception as e:
            raise Exception(f"Error loading CSV file: {str(e)}")
    
    def create_airport_dictionary(self, df: pd.DataFrame) -> Dict[str, Dict[str, str]]:
        """
        Create a dictionary of airports from the DataFrame
        
        Args:
            df: DataFrame containing airport data
            
        Returns:
            Dictionary with airport names and codes
        """
        airport_dict = {}
        
        for _, row in df.iterrows():
            airport_name = str(row['Airport']).strip() if pd.notna(row['Airport']) else ""
            airport_code = str(row['Airport Code']).strip() if pd.notna(row['Airport Code']) else ""
            
            if airport_name:
                airport_dict[airport_name.lower()] = {
                    'name': airport_name,
                    'code': airport_code if airport_code and airport_code != 'nan' else None
                }
                if airport_code and airport_code != 'nan':
                    airport_dict[airport_code.lower()] = {
                        'name': airport_name,
                        'code': airport_code
                    }
        
        return airport_dict
    
    def extract_airports_from_email(self, email_content: str, airport_dict: Dict[str, Dict[str, str]]) -> Dict:
        """
        Use LLM to extract airport names and codes from email content
        
        Args:
            email_content: The email content to analyze
            airport_dict: Dictionary of known airports
            
        Returns:
            JSON response with found airports and codes
        """
        airport_list = []
        seen_airports = set()
        
        for key, value in airport_dict.items():
            airport_info = f"{value['name']}"
            if value['code']:
                airport_info += f" ({value['code']})"
            if airport_info not in seen_airports:
                airport_list.append(airport_info)
                seen_airports.add(airport_info)
        
        airport_list_str = ", ".join(airport_list[:50])  # Limit to avoid token limits
      
        system_prompt = f"""You are an expert at extracting airport information from text. 
        Given an email content and a list of known airports, identify which airports and their codes are mentioned in the email.
        
        Known airports: {airport_list_str}
        
        Instructions:
        1. Carefully read the email content
        2. Identify any mentions of airports, cities, or airport codes
        3. Match them against the known airports list
        4. For each match, return the exact airport name and its corresponding code (if available)
        5. If only a code is mentioned, find the corresponding airport name
        6. Return the result as a JSON object with this format:
        {{
            "found_airports": [
                {{"airport_name": "Airport Name 1", "airport_code": "Code1"}},
                {{"airport_name": "Airport Name 2", "airport_code": "Code2"}}
            ]
        }}
        Be precise and only return airports that are clearly mentioned in the email content."""
        
        human_prompt = f"""Email Content:
        {email_content}
        
        Please analyze this email and extract any airport names and codes that match the known airports list. Return the result in the specified JSON format."""
        
        try:
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=human_prompt)
            ]
            
            response = self.llm.invoke(messages)
            
            try:
                response_content = response.content
                start_idx = response_content.find('{')
                end_idx = response_content.rfind('}') + 1
                
                if start_idx != -1 and end_idx != -1:
                    json_str = response_content[start_idx:end_idx]
                    result = json.loads(json_str)
                    return result
                else:
                    return {
                        "found_airports": [],
                        "error": "Could not parse JSON response",
                        "raw_response": response_content
                    }
                    
            except json.JSONDecodeError:
                return {
                    "found_airports": [],
                    "error": "JSON parsing failed",
                    "raw_response": response.content
                }
                
        except Exception as e:
            return {
                "found_airports": [],
                "error": f"LLM call failed: {str(e)}"
            }
    
    def get_airport_columns_with_y(self, df: pd.DataFrame, airport_name: str) -> List[str]:
        """
        Get column names that have 'Y' for the specified airport
        
        Args:
            df: DataFrame containing airport data
            airport_name: Name of the airport to check
            
        Returns:
            List of column names with 'Y' values
        """
        airport_row = df[df['Airport'].str.lower() == airport_name.lower()]
        
        if airport_row.empty:
            return []
        
        row = airport_row.iloc[0]
        y_columns = []
        for col in df.columns:
            if col not in ['Airport', 'Airport Code']:
                if str(row[col]).strip().upper() == 'Y':
                    y_columns.append(col)
        
        return y_columns
    
    def process_email_and_extract_info(self, email_content: str, csv_file_path: str) -> Dict:
        """
        Complete workflow: load data, extract airports, and get column info
        
        Args:
            email_content: The email content to analyze
            csv_file_path: Path to the CSV file
            
        Returns:
            Dictionary with results
        """
        try:
            df = self.load_airport_data(csv_file_path)
            airport_dict = self.create_airport_dictionary(df)
            extraction_result = self.extract_airports_from_email(email_content, airport_dict)
            
            results = {
                "extraction_result": extraction_result,
                "airport_details": []
            }
            
            if "found_airports" in extraction_result:
                for airport_info in extraction_result["found_airports"]:
                    airport_name = airport_info.get("airport_name", "")
                    if airport_name:
                        y_columns = self.get_airport_columns_with_y(df, airport_name)
                        results["airport_details"].append({
                            "airport_name": airport_name,
                            "airport_code": airport_info.get("airport_code", None),
                            "columns_with_y": y_columns
                        })
            
            return results
            
        except Exception as e:
            return {
                "error": f"Processing failed: {str(e)}",
                "extraction_result": {},
                "airport_details": []
            }

def add_to_file_start(file_path: str, text_to_add: str) -> bool:
    """Add text to the beginning of an existing file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            existing_content = f.read()
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(text_to_add + "\n\n" + existing_content)
        
        return True
    except Exception as e:
        print(f"Error: {e}")
        return False

def airport_finder(email_content: str) -> Tuple[List[str], str]:
    """
    Find airports and their codes in email content and return combined columns.
    
    Args:
        email_content: The email content to analyze
        
    Returns:
        Tuple of (combined_columns, json_output)
    """
    extractor = AirportExtractor(aws_region="us-east-1")
    csv_file_path = "Market Share Reporting 1.csv"
    
    # Load data once
    df = extractor.load_airport_data(csv_file_path)
    airport_dict = extractor.create_airport_dictionary(df)
    
    # Extract airports from email
    extraction_result = extractor.extract_airports_from_email(email_content, airport_dict)
    json_output = json.dumps(extraction_result, indent=2)
    
    # Define the base columns that are always included
    additional_columns = [
        'Month-Year', 'Airport Code', 'Airport Name', 'Company', 'Parent Company',
        'Lot Type', 'Gross Revenue (Sales)', 'Market Share (Rev Share)', 
        'CFC Per Day', 'CFC Revenue', 'Transactions', 'Transaction Days',
        'Deplaning PAX', 'Enplaned PAX', 'RAC Passengers'
    ]
    
    found_airports = extraction_result.get("found_airports")
    
    # If an airport is found, get its 'Y' columns and combine them with the additional columns
    if found_airports:
        first_airport_name = found_airports[0].get("airport_name")
        if first_airport_name:
            y_columns = extractor.get_airport_columns_with_y(df, first_airport_name)
            combined_columns = list(set(y_columns + additional_columns))
        else:
            # This case is unlikely (airport in list but no name), but for completeness
            combined_columns = additional_columns
    else:
        # If no airport is found, return only the additional columns
        combined_columns = additional_columns
            
    add_to_file_start("extracted_content.txt", json_output)
    
    return combined_columns, json_output

# Example usage with the provided email content
if __name__ == "__main__":
    email_content = """
    Subject: Travel & Meeting Coordination – DTW

    Dear Team,

    I hope this email finds you well. I wanted to discuss our upcoming business travel plans. We should arrange flights to Detroit for the meetings scheduled at John Wayne Airport next week. Additionally, please confirm availability at the Avis/Budget rental lot for our team’s transportation needs.

    Please let us know the status and coordination details as soon as possible.

    Best regards,
    John
    """
    
    combined_columns, json_output = airport_finder(email_content)
    print("Combined Columns:", combined_columns)
    print("\nExtracted Airports JSON:\n", json_output)
