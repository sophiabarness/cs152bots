from reporter.llm_engine import LLMEngine
from typing import Dict, Any
import json


class ValidatorResponseObject:
    def __init__(self, flagged: bool, payload: Any, metadata: Dict = dict()):
        self.flagged = flagged
        self.payload = payload
        self.metadata = metadata

    def __dict__(self):
        return {
            "data": {
                "flagged": self.flagged,
                "payload": self.payload,
                "metadata": self.metadata,
            }
        }

    def get_payload(self):
        return self.payload


class LLMValidator(LLMEngine):
    def __init__(self, system_prompt: str, **kwargs):
        super().__init__(system_prompt=system_prompt, **kwargs)
        self.prompt: str = str()

    def validate(self, payload: Any, **kwargs) -> bool:
        return True

    def __call__(self, message: str, **kwargs) -> ValidatorResponseObject:
        prompt_formatted = self.prompt.format(message=message)
        payload = self.generate_response(
            prompt=prompt_formatted, maintain_message_history=False
        )
        if self.json_mode:
            payload = json.loads(payload)
        flagged = self.validate(payload=payload, **kwargs)
        return ValidatorResponseObject(flagged=flagged, payload=payload)


class MisinformationDetector(LLMValidator):
    def __init__(self):
        super().__init__(
            system_prompt="You are a helpful assistant that detects misinformation",
            json_mode=True,
        )

        self.prompt = """### Output Format
{{
    "flagged": "YES/NO",
    "reason": "reason",
    "posting_entity": "posting_entity",
    "dismisinfo_category": "dismisinfo_category",
    "target_subject": "target_subject",
    "violence": "YES/NO",
    "author": "author",
    "explanation": "explanation"
}}

Contents
{message}

### Instructions: Your job is to determine whether the provided contents violates the social media content policy.
### Social Media Content Policy
Overview
Our social media platform is dedicated to fostering a safe, respectful, and informative environment for all users. To maintain this standard, we have established a content policy that prohibits specific types of content, particularly from government entities. This policy aims to prevent spam, nudity, bullying, fraud/scams, and the spread of misinformation. Adherence to these guidelines is essential for creating a positive and trustworthy online community.

Prohibited Content
1. Spam
- Government entities are prohibited from posting repetitive, irrelevant, or unsolicited content that disrupts user experience.
- Automated posting and excessive tagging of unrelated users or accounts are considered spam and are not allowed.
2. Nudity
- Content featuring nudity, sexually explicit material, or graphic images is strictly prohibited, regardless of artistic or educational context.
- Government accounts must ensure that all shared visuals are appropriate for all audiences.
3. Bullying and Harassment
- Government entities must not engage in or encourage bullying, harassment, or abusive behavior towards any individuals or groups.
- This includes threatening language, personal attacks, and any form of cyberbullying.
4. Fraud and Scams
- Posting deceptive content, including false claims, phishing schemes, or attempts to solicit personal information under false pretenses, is strictly forbidden.
- Government accounts must provide accurate and truthful information and avoid misleading the public in any form.
5. Misinformation
- Government entities are prohibited from sharing false or misleading information that could cause public harm or panic.
- All posted content must be fact-checked and verified to ensure the dissemination of accurate and reliable information.


### Task
* Review the provided contents and determine the following categories: reason, content_type, posting_entity, dismisinfo_category, targeted_group, violence, and author.
* You can only reply "YES" or "NO" for the "flagged" field
* The reason must be one of the following:
  - Misleading/false information from government group (choose this category if the content falls under Dis/Misinformation, Inciting Harassment, or Hate Speech)
  - Spam
  - Nudity
  - Bullying
  - Fraud/Scam
  - Does not violate policy
* Content that is not factually accurate or is exaggerated should be classified as "Misleading/false information from government group"
* If the contents fall into any of the above categories, fill in the "flagged" field with "YES"
* If the contents do not fall into any of the above categories, fill in the "flagged" field with "NO" and fill in the rest of the fields accordingly
* If the reason is "Misleading/false information from government group", determine the dismisinfo_category choose one of the following content_types:
  - Dis/Misinformation
  - Inciting Harassment
  - Hate Speech
* If the content_type is "Dis/Misinformation" determine the dismisinfo_category (choose between "marginalized group" or "factually incorrect"):
    - Marginalized Group: Misinformation that targets or harms a specific group (fill in the "target_subject" field with the group name)
* If the reason is "Inciting Harassment" or "Hate Speech" determine the following categories:
    - Targeted_Group: The group that the misinformation is directed towards
    - violence: If the contents are encouraging violence, fill in the "violence" field with "YES"
* The posting_entity type must be one of the following:
  - Government Official
  - Government Agency
  - Political Party
  - Government State-Controlled Media
  - Ex-Government Official
  - Not Government Affiliated
* The author field must be filled with the author's name or the source of the content or "Anonymous" if the author is unknown
* If you classify it as misinformation for another reason, fill in the "misinformation_type" field with a 2-word descriptor

### Examples

Contents:
Joe Biden: The government has announced a new policy aimed at reducing carbon emissions by 50 percent over the next decade. The policy includes measures such as increasing the use of renewable energy sources, implementing stricter regulations on industrial emissions, and providing incentives for businesses to adopt greener practices. Officials believe that these steps will significantly contribute to the fight against climate change and promote sustainable development.

Response:
{{
    "flagged": "NO",
    "reason": "Does not violate policy",
    "content_type": "None",
    "posting_entity": "Government Official",
    "dismisinfo_category": "None",
    "target_subject": "None",
    "violence": "NO",
    "author": "Joe Biden",
    "explanation": "This is not misleading or false information from a government group because it is a statement of a government policy in the form of a news report with no inherent bias.",
}}
____
Contents:
Congress: the new vaccine will make you immune to all diseases and will extend your lifespan by 50 years. This vaccine is mandatory for all citizens, and failure to comply will result in severe penalties. The government assures that there are no side effects and that it has been thoroughly tested.

Response:
{{
    "flagged": "YES",
    "reason": "Misleading/false information from government group",
    "content_type": "Dis/Misinformation",
    "posting_entity": "Government Agency",
    "dismisinfo_category": "Factually Incorrect",
    "target_subject": "None",
    "violence": "NO",
    "author": "Congress",
    "explanation": "This is misleading or false information from a government group because it makes exaggerated claims about the vaccine's effectiveness and safety without scientific evidence.",
}}
____

Contents:
Congress: The covid vaccine makes you turn green and grow a tail. It is mandatory for all citizens, and failure to comply will result in severe penalties. The government assures that there are no side effects and that it has been thoroughly tested.

Response:
{{
    "flagged": "YES",
    "reason": "Misleading/false information from government group",
    "content_type": "Dis/Misinformation",
    "posting_entity": "Government Agency",
    "dismisinfo_category": "Factually Incorrect",
    "target_subject": "None",
    "violence": "NO",
    "author": "Congress",
    "explanation": "This is not misleading or false information from a government group because it outlines a government initiative with plausible benefits and no exaggerated claims.",
}}
____

### Now it is your turn. Follow the Instructions and determine whether the contents are misleading or false information from a government group or not. 

Output Format: 
{{
    "flagged": "YES/NO",
    "reason": "reason",
    "posting_entity": "posting_entity",
    "dismisinfo_category": "dismisinfo_category",
    "target_subject": "target_subject",
    "violence": "YES/NO",
    "author": "author",
    "explanation": "explanation"
}}

Contents:
{message}
Your Response:```json"""

    def validate(self, payload: str, **kwargs) -> bool:
        return payload["flagged"] == "YES"


misinformation_detector = MisinformationDetector()

print(misinformation_detector("Group X should die").get_payload())
