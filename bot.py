from javascript import require, On, Once, AsyncTask, once, off
from dqnagent import DQNAgent
import time

mineflayer = require('mineflayer')
Vec3 = require('vec3')

bot = mineflayer.createBot({"username": "bot", "host": "localhost", "port": 3000, "version": "1.19.4", "hideErrors": False})

start= Vec3(0, 20, 0)

HORIZONTAL_MOVEMENT_ACTIONS = ["left", "right", "none"]
VERTICAL_MOVEMENT_ACTIONS = ["forward", "back", "none"]
JUMP_ACTIONS = ["jump", "none"]
SPRINT_ACTIONS = ["sprint", "none"]
HORIZONTAL_LOOK_ACTIONS = ["left", "right", "none"]
VERTICAL_LOOK_ACTIONS = ["up", "down", "none"]

input_size = 3
hm_size = len(HORIZONTAL_MOVEMENT_ACTIONS)
vm_size = len(VERTICAL_MOVEMENT_ACTIONS)
jump_size = len(JUMP_ACTIONS)
sprint_size = len(SPRINT_ACTIONS)
hl_size = len(HORIZONTAL_LOOK_ACTIONS)
vl_size = len(VERTICAL_LOOK_ACTIONS)
agent = DQNAgent(input_size, hm_size, vm_size, jump_size, sprint_size, hl_size, vl_size)

def get_state():
    return [bot.entity.position.x, bot.entity.position.y, bot.entity.position.z]

def act(hm_action, vm_action, jump_action, sprint_action, hl_action, vl_action):
    #print(hm_action, vm_action, jump_action, sprint_action, hl_action, vl_action)
    if hm_action < 2:
        bot.setControlState(HORIZONTAL_MOVEMENT_ACTIONS[hm_action], True)
    if vm_action < 2:
        bot.setControlState(VERTICAL_MOVEMENT_ACTIONS[vm_action], True)
    if jump_action == 0:
        bot.setControlState(JUMP_ACTIONS[jump_action], True)
        time.sleep(0.3)      
    if sprint_action == 0:
        bot.setControlState(SPRINT_ACTIONS[sprint_action], True)
        time.sleep(0.5)

    if hl_action == 1:
        bot.look(10, 0, timeout=5000)
    elif hl_action == 2:
        bot.look(-10, 0, timeout=5000)
    if vl_action == 1:
        bot.look(0, 2, timeout=5000)
    elif vl_action == 2:
        bot.look(0, -2, timeout=5000)
        
    bot.clearControlStates()
    
        
def reward_calc(previous_state, new_state):
    reward = 0
    movement_x = new_state[0] - previous_state[0]
    movement_z = abs(new_state[2] - previous_state[2]) 
    reward += movement_x * 10  
    reward -= movement_z * 5
    if movement_x == 0:
        reward -= 10
    return reward

def main():
    batch_size = 32
    iterations = 0
    while True:
        state = get_state()
        print(state)
        hm_action, vm_action, jump_action, sprint_action, hl_action, vl_action = agent.choose_action(state)
        act(hm_action, vm_action, jump_action, sprint_action, hl_action, vl_action)
        next_state = get_state()
        reward = reward_calc(state, next_state)
        done = False
        agent.remember(state, hm_action, vm_action, jump_action, sprint_action, hl_action, vl_action, reward, next_state, done)
        agent.replay(batch_size)
        iterations += 1
        print(f"iteration: {iterations}")
        
main()

@On(bot, "login")
def login(this):
    bot_socket = bot._client.socket
    print(f"{bot.username} logged in")

@On(bot, "messagestr")
def messagestr(this, message, messagePosition, jsonMsg, sender, verified):
    if messagePosition == "chat" and "quit" in message:
        bot.chat("HELP! SAVE ME PLEASE! DONT LET ME DIE I DONT WANT TO DIE! AHHHHHHHHHHHHHHHHHHHH")
        this.quit()
    if "run" in message:
        bot.setControlState('forward', True)

@On(bot, "kicked")
def kicked(this, reason, loggedIn):
    print(f"{bot.username} kicked for: {reason}")
    
@On(bot, "end")
def end(this,reason):
    print(f"{bot.username} disconnected")
    
    off(bot, "login", login)
    off(bot, "kicked", kicked)
    off(bot, "end", end)
    off(bot, "messagestr", messagestr)
    