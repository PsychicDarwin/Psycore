import pytest
from src.credential_manager.LocalCredentials import LocalCredentials, APICredential

def test_api_credential_creation():
    """Test creating an APICredential with secret key and optional user key."""
    # Test with only secret key
    cred1 = APICredential("test_secret")
    assert cred1.secret_key == "test_secret"
    assert cred1.user_key is None

    # Test with both secret and user key
    cred2 = APICredential("test_secret", "test_user")
    assert cred2.secret_key == "test_secret"
    assert cred2.user_key == "test_user"

def test_local_credentials_add_get():
    """Test adding and retrieving credentials from LocalCredentials."""
    # Add credential with only secret key
    LocalCredentials.add_credential("TEST_API", "test_secret")
    cred = LocalCredentials.get_credential("TEST_API")
    assert isinstance(cred, APICredential)
    assert cred.secret_key == "test_secret"
    assert cred.user_key is None

    # Add credential with both keys
    LocalCredentials.add_credential("TEST_API_2", "test_secret_2", "test_user_2")
    cred2 = LocalCredentials.get_credential("TEST_API_2")
    assert isinstance(cred2, APICredential)
    assert cred2.secret_key == "test_secret_2"
    assert cred2.user_key == "test_user_2"

def test_local_credentials_remove():
    """Test removing credentials from LocalCredentials."""
    # Add and then remove a credential
    LocalCredentials.add_credential("TEST_REMOVE", "test_secret")
    LocalCredentials.remove_credential("TEST_REMOVE")
    
    # Verify credential was removed by checking if accessing it raises KeyError
    with pytest.raises(KeyError):
        LocalCredentials.get_credential("TEST_REMOVE")

def test_local_credentials_nonexistent():
    """Test accessing non-existent credentials."""
    with pytest.raises(KeyError):
        LocalCredentials.get_credential("NONEXISTENT_API")

def test_credential_overwrite():
    """Test overwriting an existing credential."""
    # Add initial credential
    LocalCredentials.add_credential("TEST_OVERWRITE", "initial_secret", "initial_user")
    
    # Overwrite with new values
    LocalCredentials.add_credential("TEST_OVERWRITE", "new_secret", "new_user")
    
    # Verify new values
    cred = LocalCredentials.get_credential("TEST_OVERWRITE")
    assert cred.secret_key == "new_secret"
    assert cred.user_key == "new_user"

@pytest.fixture(autouse=True)
def cleanup_credentials():
    """Cleanup fixture to reset credentials after each test."""
    yield
    # Clear all test credentials after each test
    test_keys = [k for k in LocalCredentials._credentials.keys() if k.startswith("TEST_")]
    for key in test_keys:
        LocalCredentials.remove_credential(key)
