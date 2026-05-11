import keyring
import getpass

def set_key():
    service = "icmdp-agent"
    key_name = "CLAUDE_API_KEY"
    
    print(f"Setting secret for service: {service}, key: {key_name}")
    password = getpass.getpass(f"Enter your {key_name}: ")
    
    if password:
        keyring.set_password(service, key_name, password)
        print("Secret stored successfully in keyring.")
    else:
        print("No password entered. Aborting.")

if __name__ == "__main__":
    set_key()
