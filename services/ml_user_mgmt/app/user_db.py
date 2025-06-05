from pydantic import BaseModel


class UserSchema(BaseModel):
    username: str
    password: str


class UserDb:
    def __init__(self):
        # This holds our user database
        self.users = [
            UserSchema(
                username="user123",
                password="pass123",
            ),
        ]

    def check_user(self, data: UserSchema):
        for user in self.users:
            if user.username == data.username and user.password == data.password:
                return True
        return False
