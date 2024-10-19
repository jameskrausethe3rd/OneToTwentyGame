import numpy as np
import time
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

# Initialize the WebDriver
driver = webdriver.Chrome()

# Open the desired webpage
driver.get("https://fleetfoxx.github.io/1-to-20/")

class OneToTwentyGame:
    def __init__(self):
        self.start()

    def start(self):
        # Wait for the button to become clickable, with a timeout of 2 seconds
        wait = WebDriverWait(driver, 2)

        # Find start button
        start_button = wait.until(EC.element_to_be_clickable((By.CSS_SELECTOR, "#right-panel .start-button")))

        # Click the start button
        start_button.click()

        # Initialize the game with 20 empty slots (-1 indicates empty)
        self.slots = [-1] * 20
        self.used_numbers = set()  # Track used numbers
        self.current_number = int(driver.find_element(By.CLASS_NAME, "large-font").text)
        self.turn = 0
        self.done = False
        return self.get_state()

    def reset(self):
        restart_button = driver.find_element(By.CLASS_NAME, "new-game")
        restart_button.click()
        time.sleep(.05)

        # Wait for the button to become clickable, with a timeout of 10 seconds
        wait = WebDriverWait(driver, 2)

        # Find start button
        start_button = wait.until(EC.element_to_be_clickable((By.CSS_SELECTOR, "#right-panel .start-button")))

        # Click the start button
        start_button.click()
        # Reinitialize the game with 20 empty slots (-1 indicates empty)
        self.slots = [-1] * 20
        self.used_numbers = set()  # Track used numbers
        self.current_number = int(driver.find_element(By.CLASS_NAME, "large-font").text)
        self.turn = 0
        self.done = False
        return self.get_state()

    def get_current_number(self):
        number = int(driver.find_element(By.CLASS_NAME, "large-font").text)
        return number

    def get_state(self):
        # Return the current state of the game: the slots and the current number
        return {'slots': self.slots, 'current_number': self.current_number, 'turn': self.turn}

    def step(self, action):
        """
        action: index (0 to 19) where the current number should be placed
        Returns:
            next_state, reward, done
        """
        if not self.is_valid_action(action):
            # Invalid action (slot already taken or out of order), negative reward
            reward = -10
        else:
            # Place the number and calculate reward
            self.slots[action] = self.current_number
            slot_element = driver.find_element(By.XPATH, f"/html/body/div/div[1]/div[1]/div[{action + 1}]/button")
            slot_element.click()
            reward = self.calculate_reward(action)

        # Check if game is over
        if self.turn == 20 or not self.has_valid_spots():
            self.done = True
        else:
            self.check_if_done()
            self.current_number = self.get_current_number()
            self.turn += 1

        return self.get_state(), reward, self.done

    def is_valid_action(self, action):
        # Check if the action is valid: slot should be empty and placement should follow rules
        if self.slots[action] != -1:
            return False
        
        valid_spots = self.get_valid_spots()
        return action in valid_spots

    def get_valid_spots(self):
        valid_slots = []

        # Find the slot elements
        slot_elements = driver.find_elements(By.XPATH, "/html/body/div/div[1]/div[1]/*")
            
        for idx, slot in enumerate(slot_elements):
            # Find the second element in the div (index 1 because index 0 is the span to be ignored)
            second_element = slot.find_elements(By.XPATH, './*')[1]

            if second_element.tag_name == "button":
                # Check if the button has the 'disabled' attribute
                if not second_element.get_attribute('disabled'):
                    # If it's not disabled, add the index to valid_slots
                    valid_slots.append(idx)
            elif second_element.tag_name == "span":
                # Check if the span has the class "locked-in"
                if "locked-in" not in second_element.get_attribute('class'):
                    valid_slots.append(idx)

        return valid_slots

    def has_valid_spots(self):
        # Check if there are any empty spots available
        return len(self.get_valid_spots()) > 0

    def check_if_done(self):
        try:
            game_end_element = driver.find_element(By.CSS_SELECTOR, ".label.red")
            return game_end_element is not None  # Return True if the element is found
        except:
            return False  # Return False if the element is not found within the timeout

    def calculate_reward(self, action):
        """
        Calculate the reward based on how well the number is placed.
        Higher rewards for placing the number in the correct position.
        """
        reward = 0

        # Reward for correct sorting
        if (action == 0 or self.slots[action - 1] <= self.current_number) and \
           (action == 19 or self.slots[action + 1] >= self.current_number or self.slots[action + 1] == -1):
            reward += 1

        if (action == self.current_number // 50):
            reward += 10

        return reward
