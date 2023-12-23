import os
import pickle

from dotenv import load_dotenv
import telebot

from inference.model_inference import process
from models.model import Model
from inference.docs import HELP_DOC, dict_to_doc, PROCESS_DOC, REQUEST_HELP, HELLO

load_dotenv()
BOT_TOKEN = os.environ.get("BOT_TOKEN")
MODEL_PATH = os.environ.get("MODEL_PATH")

xgb = pickle.load(open(MODEL_PATH, "rb"))
model = Model(xgb)

bot = telebot.TeleBot(BOT_TOKEN)


@bot.message_handler(commands=["start"])
def start(message):
    bot.reply_to(message, HELLO)


@bot.message_handler(commands=["help"])
def help(message):
    bot.reply_to(message, HELP_DOC)


@bot.message_handler(commands=["features"])
def features(message):
    bot.reply_to(message, dict_to_doc(model.get_features_dict()))


@bot.message_handler(commands=["process"])
def process_handler(message):
    send = bot.send_message(
        message.chat.id,
        PROCESS_DOC,
    )
    bot.register_next_step_handler(send, process_following)


def process_following(message):
    bot.send_message(message.chat.id, process(model, message.text))


@bot.message_handler()
def other(message):
    bot.reply_to(message, REQUEST_HELP)


print("Bot started")
bot.infinity_polling()
