import os

from dotenv import load_dotenv
import telebot

from inference.model_inference import process

load_dotenv()
BOT_TOKEN = os.environ.get("BOT_TOKEN")

bot = telebot.TeleBot(BOT_TOKEN)


@bot.message_handler(commands=["help"])
def help(message):
    bot.reply_to(
        message,
        "/process -- ввод данных для расчета;",
    )


@bot.message_handler(commands=["process"])
def process_handler(message):
    send = bot.send_message(
        message.chat.id, "Введите переменные в формате key: value через запятую.\nНапример:\n a: 1, b: 2"
    )
    bot.register_next_step_handler(send, process_following)


def process_following(message):
    bot.send_message(message.chat.id, process(message.text))


print("Bot started")
bot.infinity_polling()
