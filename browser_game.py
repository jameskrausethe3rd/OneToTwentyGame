from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
import torch
from model import DQNModel  # Ensure this matches the class name in your model.py

# Initialize the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialize the PyTorch model
model = DQNModel(22, 20).to(device)  # Ensure model is moved to the correct device
model.load_state_dict(torch.load('trained_model.pth', map_location=device))
model.eval()

# Initialize the WebDriver
driver = webdriver.Chrome()

# Open the desired webpage
driver.get("https://fleetfoxx.github.io/1-to-20/")

# Wait for the button to become clickable, with a timeout of 10 seconds
wait = WebDriverWait(driver, 10)

# Function to wait until the game ends (when a specific element becomes visible)
def wait_for_game_end():
    try:
        game_end_element = driver.find_element(By.CSS_SELECTOR, ".label.red")
        return game_end_element is not None  # Return True if the element is found
    except:
        return False  # Return False if the element is not found within the timeout

# Function to restart the game by clicking the restart button
def restart_game():
    time.sleep(.1)
    restart_button = driver.find_element(By.CLASS_NAME, "new-game")
    restart_button.click()

# Function to predict the best div to click using the PyTorch model
def predict_div_to_click(currentNumber, slots, turn):
    # Replace None in slots with a default value (e.g., -1)
    slots = [slot if slot is not None else -1 for slot in slots]  # Or any default value

    # Prepare the input tensor
    state_array = slots + [int(currentNumber), int(turn)]
    state_tensor = torch.tensor([state_array], dtype=torch.float32).to(device)

    # Pass the input through the model to get the prediction
    with torch.no_grad():  # Disable gradient calculation for inference
        output = model(state_tensor)

    # Get the index of the div to click (highest predicted value)
    predicted_div_index = torch.argmax(output, dim=1).item()  # Indices should match div numbers (0-19)
    
    return predicted_div_index + 1  # Add 1 to match div numbers (1-20)

def get_current_slots():
    # This function should return the current state of the slots as a list
    slots = []
    for i in range(1, 21):  # Assuming slots are numbered from 1 to 20
        try:
            # Locate the button or span element in the slot
            slot_element = driver.find_element(By.XPATH, f"/html/body/div/div[1]/div[1]/div[{i}]/button | /html/body/div/div[1]/div[1]/div[{i}]/span[2]")
            
            # Check if the element is a button (enabled) or a span (locked-in)
            if slot_element.tag_name == 'button':
                slots.append(None)  # No number placed in this slot
            elif slot_element.tag_name == 'span':
                # Retrieve the number from the span
                number = int(slot_element.text)
                slots.append(number)
        except Exception as e:
            slots.append(None)  # Slot is empty or not found
    return slots

# Main loop
turn = 0  # Initialize turn counter

# Main loop
while True:
    try:
        # Find start button
        start_button = wait.until(EC.element_to_be_clickable((By.CSS_SELECTOR, "#right-panel .start-button")))

        # Click the start button
        start_button.click()

        turn = 0
        while True:
            # Get the current number
            currentNumberElement = driver.find_element(By.CLASS_NAME, "large-font")
            currentNumber = currentNumberElement.text
            
            # Get the current slots
            slots = get_current_slots()

            # Get the current turn (increment or track as per your game logic)
            turn += 1

            # Pass the current number, slots, and turn to the model to get the div to click
            div_to_click = predict_div_to_click(currentNumber, slots, turn)

            # Click the predicted div
            button = driver.find_element(By.XPATH, f"/html/body/div/div[1]/div[1]/div[{div_to_click}]/button")
            button.click()
            time.sleep(.1)

            # Check if the game end element is visible
            if wait_for_game_end():
                print("Game ended, restarting...")
                break

        # Restart the game
        restart_game()

    except Exception as e:
        print(f"An error occurred: {e}")
        break 