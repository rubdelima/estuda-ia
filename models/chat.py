from abc import ABC, abstractmethod

class AbstractChatModel(ABC):

    @abstractmethod
    def generate_response(self, user_input: str) -> str:
        pass
    
    