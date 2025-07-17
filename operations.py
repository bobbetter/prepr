from mem0 import MemoryClient
import os
from llama_parse import LlamaParse
import re
from enum import Enum

class ContextType(Enum):
    CV = "cv_text"
    JOB_DESCRIPTION = "job_description"
    INTERVIEWER_INFO = "interviewer_info"

class PreprOperations:
    def __init__(self, user_id: str):
        self.user_id = user_id
        self.memory_client = self.get_memory_client()
        self.parser = self.get_parser()

    def get_memory_client(self) -> MemoryClient:
        mem0_api_key = os.getenv("MEM0_API_KEY")
        if not mem0_api_key:
            raise ValueError("MEM0_API_KEY not found in environment variables")
        return MemoryClient(api_key=mem0_api_key)
    
    def get_parser(self) -> LlamaParse:
        llama_parse_api_key = os.getenv("LLAMA_PARSE_API_KEY")
        if not llama_parse_api_key:
            raise ValueError("LLAMA_PARSE_API_KEY not found in environment variables")
        return LlamaParse(api_key=llama_parse_api_key, result_type="text", verbose=False)
    
    def parse_cv(self, cv_file_path: str) -> str:
        """Parse CV PDF using LlamaParse and anonymize the content."""
        if not os.path.exists(cv_file_path):
            return f"❌ Error: CV file not found: {cv_file_path}"
            
        try:
            print(f"🔄 Parsing CV: {cv_file_path}")
            
            # Parse the PDF
            documents = self.parser.load_data(cv_file_path)
            
            # Extract text from parsed documents
            cv_text = ""
            for doc in documents:
                cv_text += doc.text + "\n"
                
            # Anonymize the text
            anonymized_cv_text = self.anonymize_text(cv_text.strip())
            
            # Store in context
            return anonymized_cv_text
            
        except Exception as e:
            return f"❌ Error parsing CV: {str(e)}"

    def _filter_memories(self, memories, context_type: ContextType):
        return [x for x in memories if x["metadata"] == {'context_type': context_type.value}]

    def load_context_from_memory(self, context_type: ContextType):
        """Restore specific context item from memory."""
        try:
            # Search for stored context items
            all_memories = self.memory_client.get_all(
                user_id=self.user_id
            )
            # Helper function needed, as SDK filter doesnt work
            filtered_memories = self._filter_memories(all_memories, context_type)

            if len(filtered_memories) == 0:
                print(f"📝 No {context_type} found in memory")
                return None
            elif len(filtered_memories) > 1:
                print(f"📝 Multiple {context_type} found in memory, using the most recent one")
                filtered_memories = filtered_memories[-1]
            else:
                filtered_memories = filtered_memories[0]
                print(f"📝 Found {context_type} in memory")
            return filtered_memories["memory"]
        except Exception as e:
            print(f"Error loading {context_type} from memory: {e}")
            return None
    
    def store_context_in_memory(self, content: str, context_type: ContextType):
        """Store specific context items in memory with metadata."""
        try:
            messages = [
                {"role": "user", "content": content}
            ]
            self.memory_client.add(
                messages,
                user_id=self.user_id,
                metadata={"context_type": context_type.value},
                infer=False
            )
            print(f"✅ Stored {context_type} in memory")
        except Exception as e:
            print(f"❌ Error storing {context_type} in memory: {str(e)}")

    def anonymize_text(self, text: str) -> str:
        """Anonymize company names and personal information in the text."""
        anonymized_text = text
        
        # Common patterns for names (basic anonymization)
        name_patterns = [
            r'\b[A-Z][a-z]+\s+[A-Z][a-z]+\b',  # First Last name pattern
            r'\b[A-Z][a-z]+\s+[A-Z]\.\s+[A-Z][a-z]+\b',  # First M. Last pattern
        ]
        
        for pattern in name_patterns:
            anonymized_text = re.sub(pattern, '[NAME]', anonymized_text)
        
        # Common company patterns and known tech companies
        company_patterns = [
            r'\bGoogle\b', r'\bMicrosoft\b', r'\bApple\b', r'\bAmazon\b', r'\bMeta\b',
            r'\bFacebook\b', r'\bNetflix\b', r'\bTesla\b', r'\bUber\b', r'\bAirbnb\b',
            r'\bStripe\b', r'\bSpotify\b', r'\bSlack\b', r'\bZoom\b', r'\bDropbox\b',
            r'\bSalesforce\b', r'\bOracle\b', r'\bIBM\b', r'\bIntel\b', r'\bNVIDIA\b',
            r'\bAdobe\b', r'\bTwitter\b', r'\bLinkedIn\b', r'\bSquare\b', r'\bPayPal\b', 
            r'\bSiriusXM\b', r'\bPandora\b'
        ]
        
        for pattern in company_patterns:
            anonymized_text = re.sub(pattern, '[COMPANY]', anonymized_text, flags=re.IGNORECASE)
        
        # Generic company patterns
        company_suffixes = [
            r'\b\w+\s+Inc\.?\b', r'\b\w+\s+LLC\.?\b', r'\b\w+\s+Corp\.?\b',
            r'\b\w+\s+Corporation\b', r'\b\w+\s+Limited\b', r'\b\w+\s+Ltd\.?\b',
            r'\b\w+\s+Company\b', r'\b\w+\s+Co\.?\b'
        ]
        
        for pattern in company_suffixes:
            anonymized_text = re.sub(pattern, '[COMPANY]', anonymized_text, flags=re.IGNORECASE)
        
        # Email addresses
        anonymized_text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[EMAIL]', anonymized_text)
        
        # Phone numbers
        phone_patterns = [
            r'\b\d{3}-\d{3}-\d{4}\b',  # 123-456-7890
            r'\b\(\d{3}\)\s*\d{3}-\d{4}\b',  # (123) 456-7890
            r'\b\d{3}\.\d{3}\.\d{4}\b',  # 123.456.7890
            r'\b\+\d{1,3}\s*\d{3}\s*\d{3}\s*\d{4}\b'  # +1 123 456 7890
        ]
        
        for pattern in phone_patterns:
            anonymized_text = re.sub(pattern, '[PHONE]', anonymized_text)
        
        return anonymized_text
    

def main():
    operations = PreprOperations(user_id="interview_prep_agent")
    # cv_text = operations.parse_cv("cv.pdf")
    operations.store_context_in_memory("test", ContextType.CV)
    # print(cv_text)

if __name__ == "__main__":
    main()