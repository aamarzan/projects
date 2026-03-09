import time
import sys
import random
import os
import shutil

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def print_centered(text, vertical_offset=0):
    cols, lines = shutil.get_terminal_size()
    horizontal_padding = (cols - len(text)) // 2
    vertical_padding = max(0, (lines // 2) + vertical_offset)
    print('\n' * vertical_padding + ' ' * horizontal_padding + text, end='', flush=True)

def print_romantic_message():
    hearts = ['❤️', '💖', '💘', '💝', '💗', '💓', '💞']
    colors = ['\033[91m', '\033[95m', '\033[93m', '\033[96m']
    
    # Initial centered message
    message = "Moni... there's something I need to tell you..."
    clear_screen()
    for i in range(len(message) + 1):
        clear_screen()
        print_centered(message[:i])
        time.sleep(0.1)
    time.sleep(1)
    
    # Big centered "I LOVE YOU" reveal
    love_msg = "I LOVE YOU MONI! ❤️"
    for i in range(len(love_msg)):
        clear_screen()
        colored_char = random.choice(colors) + love_msg[i] + '\033[0m'
        print_centered(love_msg[:i] + colored_char)
        time.sleep(0.2)
    
    # Centered hearts animation
    for _ in range(15):
        clear_screen()
        heart = random.choice(hearts)
        colored_heart = random.choice(colors) + heart + '\033[0m'
        print_centered(colored_heart)
        time.sleep(0.3)
    
    # Final centered message that stays
    final_msg = "Will you be mine forever? 💍"
    clear_screen()
    for i in range(len(final_msg)):
        print_centered(final_msg[:i+1])
        time.sleep(0.1)
    
    # Keep Moni visible at the end
    time.sleep(2)
    clear_screen()
    print_centered("❤️ Moni ❤️", 0)
    time.sleep(3)

print_romantic_message()