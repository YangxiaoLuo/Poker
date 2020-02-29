from enum import Enum


class Action(Enum):
    '''
    A Enum that contains all availabel actions in ToyPoker
    '''
    FOLD = 'fold'
    CHECK = 'check'
    CALL = 'call'
    RAISE = 'raise'

# if __name__ == "__main__":
#     print(Action.FOLD.value)