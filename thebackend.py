from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.keys import Keys
from urllib.request import urlretrieve
import time
import csv

url = 'https://www.thebackend.io/'
driver = webdriver.Chrome('/Users/USER/Desktop/벤처/chromedriver-win32/chromedriver')
driver.get(url)
time.sleep(3)

driver.find_element_by_xpath('//*[@id="header_inner"]/div[2]/nav/ul/li[2]/a').click()
time.sleep(1.5)

body = driver.find_element_by_tag_name('body')
for i in range(20):
    body.send_keys(Keys.PAGE_DOWN)
    time.sleep(0.5)

image_list = list()
name_list = list()
company_list = list()
download_list = list()
age_list = list()
update_list = list()
genre_list = list()
game_list = list()

idx = 1
while(1):

    image = driver.find_elements_by_xpath('//*[@id="showcase-list"]/li['+str(idx)+']/img[1]')
    for img in image: image_list.append(img.get_attribute('src'))
    name = driver.find_elements_by_xpath('//*[@id="showcase-list"]/li['+str(idx)+']/h4')
    for na in name: name_list.append(na.text)
    company = driver.find_elements_by_xpath('//*[@id="showcase-list"]/li['+str(idx)+']/h2')
    for comp in company: company_list.append(comp.text)

    lx = driver.find_element_by_xpath('//*[@id="showcase-list"]/li['+str(idx)+']/div/div/a[1]')
    link = lx.get_attribute("href")
    driver.execute_script('window.open("https://naver.com");')
    time.sleep(0.5)

    driver.switch_to.window(driver.window_handles[-1])
    driver.get(link)
    time.sleep(2)

    xpath1 = '//*[@id="yDmH0d"]/c-wiz[2]/div/div/div[2]/div[1]/div/div/c-wiz/div[2]/div[2]/div/div/div[2]/div[2]/span/span'
    xp1 = driver.find_elements_by_xpath(xpath1)
    xpath2 = '//*[@id="yDmH0d"]/c-wiz[2]/div/div/div[2]/div[1]/div/div/c-wiz/div[2]/div[2]/div/div/div[3]/div[2]/span/span'
    xp2 = driver.find_elements_by_xpath(xpath2)

    if xp1: x, y = 1, 2
    else:
        if xp2: x, y = 2, 3
        else: x, y = 2, 4
    download_xpath = '//*[@id="yDmH0d"]/c-wiz[2]/div/div/div[2]/div[1]/div/div/c-wiz/div[2]/div[2]/div/div/div['+str(x)+']/div[1]'
    age_xpath = '//*[@id="yDmH0d"]/c-wiz[2]/div/div/div[2]/div[1]/div/div/c-wiz/div[2]/div[2]/div/div/div['+str(y)+']/div[2]/span/span'
    update_xpath = '//*[@id="yDmH0d"]/c-wiz[2]/div/div/div[2]/div[2]/div/div[1]/div[1]/c-wiz[2]/div/section/div/div[2]/div/div[2]'

    download = driver.find_elements_by_xpath(download_xpath)
    for down in download: download_list.append(down.text)
    if not download:
        idx += 1
        image_list.pop()
        name_list.pop()
        company_list.pop()
        time.sleep(0.5)
        driver.close()
        driver.switch_to.window(driver.window_handles[-1])
        continue
    download_list[0] = download_list[0].replace("+", "").replace("천", "000").replace("만", "0000")
    download_list[0] = int(download_list[0])

    age = driver.find_elements_by_xpath(age_xpath)
    for ag in age: age_list.append(ag.text)
    update = driver.find_elements_by_xpath(update_xpath)
    for ud in update: update_list.append(ud.text)

    i = 1
    while 1:
        genre_xpath = '//*[@id="yDmH0d"]/c-wiz[2]/div/div/div[2]/div[2]/div/div[1]/div[1]/c-wiz[2]/div/section/div/div[3]/div['+f'{i}'+']/div/span'
        genre = driver.find_elements_by_xpath(genre_xpath)
        if not genre:
            i += 1
            continue
        else:
            for ge in genre: genre_list.append(ge.text)
            break

    game_list.append([image_list.pop(), name_list.pop(), company_list.pop(), download_list.pop(), age_list.pop(), update_list.pop(), genre_list.pop(), link])
    
    time.sleep(0.5)
    driver.close()
    driver.switch_to.window(driver.window_handles[-1])

    idx += 1
    if idx == 51: break

time.sleep(0.5)
driver.close()
game_list.sort(key = lambda x : (x[6], -x[3]))

with open('thebackend.csv', 'w', newline='') as f:
    write = csv.writer(f)
    write.writerows(game_list)
