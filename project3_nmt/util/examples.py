import dataclasses
import json

from dataclasses import dataclass
from typing import List, Optional

import logging

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%Sgp',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class CFExample(object):
    guid: str
    input_ids: List[int]
    category_id: int

    def to_dict(self):
        return dataclasses.asdict(self)

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2) + "\n"


# object 상속시 생성자 pass 가능
## 멤머변수(클래스변수) 는 나중에 사용하는 애들같음.
@dataclass
class NMTExample(object):
    guid: str
    input_ids: List[int]
    trg_ids: Optional[List[int]] = None

    def to_dict(self):
        return dataclasses.asdict(self)

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2) + "\n"
