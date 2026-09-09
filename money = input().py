money1 = int(input())
taxOwed = [0]


def lowerInc(money1, money2):
    if money1 > 57375:
        taxOwed.append(57375 * 0.145)
        money2 = money1 - 57375
    else:
        taxOwed.append(money1 * 0.145)
        money2 = 0

    return money2

def lowMidInc(money2, money3):
    if money2 > 57375:
        taxOwed.append(57375 * 0.205)
        money3 = money2 - 57375
    else:
        taxOwed.append(money2 * 0.205)
        money3 = 0

    return money3

def midInc(money3, money4):
    if money3 > 63132:
        taxOwed.append(63132 * 0.26)
        money4 = money3 - 63132
    else:
        taxOwed.append(money3 * 0.26)
        money4 = 0

    return money4

def midHighInc(money4, money5):
    if money4 > 75532:
        taxOwed.append(75532 * 0.29)
        money5 = money4 - 75532
    else:
        taxOwed.append(money4 * 0.29)
        money5 = 0

    return money5

def highInc(money5, money6):
    money6 = 0
    taxOwed.append(money5 * 0.33)

    return money6

money2 = 0
money3 = 0
money4 = 0
money5 = 0
money6 = 0

money2 = lowerInc(money1, money2)
money3 = lowMidInc(money2, money3)
money4 = midInc(money3, money4)
money5 = midHighInc(money4, money5)
money6 = highInc(money5, money6)

print("tax: " + str(sum(taxOwed)))
print("final money: " + str(money1 - sum(taxOwed)))

'''
gross code
'''