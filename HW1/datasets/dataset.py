from typing import List, Dict
import nltk
import torch
from nltk import word_tokenize
from torch.utils.data import Dataset
# nltk.download('punkt')
from util.utils import Vocab


class SeqClsDataset(Dataset):
    labels = [
        "accept_reservations",
        "account_blocked",
        "alarm",
        "application_status",
        "apr",
        "are_you_a_bot",
        "balance",
        "bill_balance",
        "bill_due",
        "book_flight",
        "book_hotel",
        "calculator",
        "calendar",
        "calendar_update",
        "calories",
        "cancel",
        "cancel_reservation",
        "car_rental",
        "card_declined",
        "carry_on",
        "change_accent",
        "change_ai_name",
        "change_language",
        "change_speed",
        "change_user_name",
        "change_volume",
        "confirm_reservation",
        "cook_time",
        "credit_limit",
        "credit_limit_change",
        "credit_score",
        "current_location",
        "damaged_card",
        "date",
        "definition",
        "direct_deposit",
        "directions",
        "distance",
        "do_you_have_pets",
        "exchange_rate",
        "expiration_date",
        "find_phone",
        "flight_status",
        "flip_coin",
        "food_last",
        "freeze_account",
        "fun_fact",
        "gas",
        "gas_type",
        "goodbye",
        "greeting",
        "how_busy",
        "how_old_are_you",
        "improve_credit_score",
        "income",
        "ingredient_substitution",
        "ingredients_list",
        "insurance",
        "insurance_change",
        "interest_rate",
        "international_fees",
        "international_visa",
        "jump_start",
        "last_maintenance",
        "lost_luggage",
        "make_call",
        "maybe",
        "meal_suggestion",
        "meaning_of_life",
        "measurement_conversion",
        "meeting_schedule",
        "min_payment",
        "mpg",
        "new_card",
        "next_holiday",
        "next_song",
        "no",
        "nutrition_info",
        "oil_change_how",
        "oil_change_when",
        "order",
        "order_checks",
        "order_status",
        "pay_bill",
        "payday",
        "pin_change",
        "play_music",
        "plug_type",
        "pto_balance",
        "pto_request",
        "pto_request_status",
        "pto_used",
        "recipe",
        "redeem_rewards",
        "reminder",
        "reminder_update",
        "repeat",
        "replacement_card_duration",
        "report_fraud",
        "report_lost_card",
        "reset_settings",
        "restaurant_reservation",
        "restaurant_reviews",
        "restaurant_suggestion",
        "rewards_balance",
        "roll_dice",
        "rollover_401k",
        "routing",
        "schedule_maintenance",
        "schedule_meeting",
        "share_location",
        "shopping_list",
        "shopping_list_update",
        "smart_home",
        "spelling",
        "spending_history",
        "sync_device",
        "taxes",
        "tell_joke",
        "text",
        "thank_you",
        "time",
        "timer",
        "timezone",
        "tire_change",
        "tire_pressure",
        "todo_list",
        "todo_list_update",
        "traffic",
        "transactions",
        "transfer",
        "translate",
        "travel_alert",
        "travel_notification",
        "travel_suggestion",
        "uber",
        "update_playlist",
        "user_name",
        "vaccines",
        "w2",
        "weather",
        "what_are_your_hobbies",
        "what_can_i_ask_you",
        "what_is_your_name",
        "what_song",
        "where_are_you_from",
        "whisper_mode",
        "who_do_you_work_for",
        "who_made_you",
        "yes",
    ]
    def __init__(
        self,
        data: List[Dict],
        vocab: Vocab,
        max_len: int,
        mode: str = 'train'
    ): 
        self.mode = mode
        self.data = data
        self.vocab = vocab
        self.label_list = self.labels
        self.label_mapping = {label: i for i, label in enumerate(self.labels)}
        self._idx2label = {idx: intent for intent, idx in self.label_mapping.items()}
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index) -> Dict:
        instance = self.data[index]
        text = instance['text']
        word_list = nltk.word_tokenize(text)
        if self.mode != 'train':
            return {'data': word_list, 'target': instance['id']}
        intent = instance['intent']
        label = self.label2idx(intent)
        return {'data': word_list, 'target': label}

    @property
    def num_classes(self) -> int:
        return len(self.label_mapping)

    def collate_fn(self, samples: List[Dict]) -> List:

        data = [item['data'] for item in samples]  # just form a list of tensor
        target = [item['target'] for item in samples] 
        data = self.vocab.encode_batch(data)
        if self.mode == 'train':
            target = torch.LongTensor(target)
        data = torch.LongTensor(data)
        return [data, target]

    def label2idx(self, label: str):
        return self.label_mapping[label]

    def idx2label(self, idx: int):
        return self._idx2label[idx]


class SeqTaggingClsDataset(SeqClsDataset):
    ignore_idx = -100
    tags = (
        "O",
        "B-date",
        "I-date",
        "B-time",
        "I-time",
        "B-people",
        "I-people",
        "B-first_name",
        "B-last_name",
    )

    def __init__(
        self,
        data: List[Dict],
        vocab: Vocab,
        max_len: int,
        mode: str = 'train'
    ): 
        self.mode = mode
        self.data = data
        self.vocab = vocab
        self.label_list = self.labels
        self.label_mapping = {label: i for i, label in enumerate(self.tags)}
        self._idx2label = {idx: intent for intent, idx in self.label_mapping.items()}
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index) -> Dict:
        instance = self.data[index]
        tokens = instance['tokens']
        word_idx = torch.LongTensor([self.vocab.token_to_id(word) for word in tokens])
        # print(word_idx)
        if self.mode != 'train':
            # print(instance['id'])
            return [word_idx, instance['id']]
        tags = instance['tags'] 
        label_list = torch.LongTensor([self.label2idx(intent)for intent in tags])
        
        
        return [word_idx , label_list]

    @property
    def num_classes(self) -> int:
        return len(self.label_mapping)

    def label2idx(self, label: str):
        return self.label_mapping[label]

    def idx2label(self, idx: int):
        return self._idx2label[idx]
