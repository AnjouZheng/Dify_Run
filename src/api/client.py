import asyncio
import aiohttp
import json
import logging
from typing import Dict, Any, Optional
from pydantic import BaseModel, EmailStr, constr

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class EmailRequest(BaseModel):
    email_content: constr(min_length=1, max_length=10000)
    sender: Optional[EmailStr] = None
    subject: Optional[str] = None

class SpamClassificationClient:
    def __init__(
        self, 
        base_url: str = 'http://localhost:8000/predict', 
        timeout: int = 10
    ):
        self.base_url = base_url
        self.timeout = timeout
        self.session = None

    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    async def predict_spam(self, email: EmailRequest) -> Dict[str, Any]:
        try:
            async with self.session.post(
                self.base_url, 
                json=email.dict(),
                timeout=self.timeout
            ) as response:
                response.raise_for_status()
                result = await response.json()
                return result
        except aiohttp.ClientError as e:
            logger.error(f"API request failed: {e}")
            raise
        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing error: {e}")
            raise

async def main():
    test_emails = [
        EmailRequest(
            email_content="Buy now! Special discount for limited time!",
            sender="marketing@spam.com",
            subject="Exclusive Offer"
        ),
        EmailRequest(
            email_content="Hi, can we discuss the project details?",
            sender="colleague@company.com",
            subject="Project Meeting"
        )
    ]

    async with SpamClassificationClient() as client:
        for email in test_emails:
            try:
                result = await client.predict_spam(email)
                print(f"Email: {email.subject}")
                print(f"Spam Probability: {result.get('spam_probability', 'N/A')}")
                print(f"Classification: {result.get('is_spam', 'N/A')}\n")
            except Exception as e:
                logger.error(f"Prediction error: {e}")

if __name__ == "__main__":
    asyncio.run(main())