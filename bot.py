from javascript import require, On, Once, AsyncTask, once, off

mineflayer = require('mineflayer')

bot = mineflayer.createBot({"username": "bot", "host": "localhost", "port": 3000, "version": "1.19.4", "hideErrors": False})

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
    