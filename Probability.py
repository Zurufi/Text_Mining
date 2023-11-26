# Ali Alzurufi
# Professor Lauren
# Date: September 10 2023
# MCS 5223: Text Mining and Data Analytics

""" Description: This program will allow the user to input a series of numbers display
    the mean, median, mode, variance, and standard deviation. The user will also be allowed to
    specify the number of dice rolls for a simulation, which will display the experimental probabilities. """

import numpy as np
import collections


# Compute the probability
def event_probability(event_outcomes, sample_space):
    probability = (event_outcomes / sample_space) * 100
    return round(probability, 1)


# Compute the mean
def mean(data):
    return np.mean(data)


# Compute the median
def median(data):
    return np.median(data)


# Compute the mode
def mode(data):
    data_count = collections.Counter(data)
    data_count_dict = dict(data_count)

    max_value = max(data_count.values())

    mode = [num for num, freq in data_count_dict.items() if freq == max_value]

    return mode if max_value > 1 else None


# Compute the variance
def variance(data):
    return np.var(data)


# Compute the standard deviation
def standard_deviation(data):
    return np.std(data)


# Simulate the dice rolls and compute probability of rolling a 3 or an odd number
def simulate_dice_rolls(number_of_dice_rolls):
    probabilities = np.random.randint(1, 7, size=number_of_dice_rolls)

    probability_of_3 = np.mean(probabilities == 3)

    probability_of_odds = np.mean(probabilities % 2 == 1)

    return probability_of_3, probability_of_odds


# Main function for user input for a dataset and the number of dice rolls
# This function will display the number of dice rolls and probabilities
# This fuction will also display the mean, median, mode, variance, and standard deviation
def main():
    data_set = input("Enter 10 numbers seperated by a space: ").split()
    print()
    data_set = list(map(int, data_set))

    num_dice_rolls = int(
        input("Enter the number of times you would like to roll the die: ")
    )
    print()
    probability_of_3, probability_of_odds = simulate_dice_rolls(num_dice_rolls)

    print(f"Probability of rolling a 3: \n{probability_of_3}\n")

    print(f"Probability of rolling an odd number: \n{probability_of_odds}\n")

    print(f"Mean: \n{mean(data_set)}\n")

    print(f"Median: \n{median(data_set)}\n")

    print(f"Mode: \n{mode(data_set)}\n")

    print(f"Variance: \n{variance(data_set)}\n")

    print(f"Standard Deviation: \n{standard_deviation(data_set)}\n")


main()
