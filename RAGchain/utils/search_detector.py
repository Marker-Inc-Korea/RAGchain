from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema import BaseChatMessageHistory


class SearchDetector:
    """
    This class is used to detect the search intent from the given chat dialogues.
    If the question is a search intent, this class will return True.
    If not, this class will return False.
    """

    def __init__(self):
        last_system_message = """
        Now, I want you to determine whether the user\'s questions are search intent or not.
        You can find hotel or restaurant review and faq data at the database.
        Do you want to search the database, or not? Please make a choice between yes or no.
        
        For example, there are list of user\'s last question and your choice:
        
        <Example 1>
        User: I have back issues. Does this place have comfortable beds?
        Choice (yes/no): yes
        
        <Example 2>
        User: That sounds great, could I get their address and phone number?
        Choice (yes/no): no
        
        <Example 3>
        User: As long as La Tasca is located in the centre of town and is moderately priced. If it is, yes that would be great. I need a reservation.
        Choice (yes/no): no
        
        <Example 4>
        User: Does The Archway House have fast and reliable wi-fi so I can connect to the Internet for work?
        Choice (yes/no): yes
        
        <Example 5>
        User: Yes, but are the staff there friendly, polite, and responsive?
        Choice (yes/no): yes
        """

        # TODO: Use different prompt for final version
        self.prompt = ChatPromptTemplate.from_messages([
            ('system', 'I want to act as a chatbot AI for booking hotel or restaurant. '
                       'You need to answer customer\'s questions nicely.'),
            MessagesPlaceholder(variable_name="history"),
            ('system', last_system_message),
            ('ai', 'Choice (yes/no): ')
        ])
        self.model = ChatOpenAI()

    def detect(self, history: BaseChatMessageHistory):
        """
        :param history: instance of BaseChatMessageHistory. Only use human and ai messages for detection.
        """
        pass
