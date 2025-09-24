from app.common.Settings import Settings

def test_Settings():

    server = Settings()
    assert server.epsilon_decay == 1.0