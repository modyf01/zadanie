import random
import collections
import csv


def save_to_csv(data, filename):
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Count', 'Average Winnings'])
        for key in sorted(data.keys()):
            writer.writerow([key, data[key]])


def save_difference_to_csv(insurance_data, no_insurance_data, filename):
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Count', 'Difference in Average Winnings'])
        for key in sorted(insurance_data.keys()):
            difference = insurance_data[key] - no_insurance_data[key]
            writer.writerow([key, difference])


USTON_SS_VALUES = {
    '2': 2, '3': 2, '4': 2, '5': 3, '6': 2,
    '7': 1, '8': 0, '9': -1, '10': -2,
    'J': -2, 'Q': -2, 'K': -2, 'A': -2
}


class Deck:
    def __init__(self, num_decks=6):
        self.num_decks = num_decks
        self.shuffle()

    def deal(self):
        return self.cards.pop()

    def shuffle(self):
        self.cards = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A'] * 4 * self.num_decks
        random.shuffle(self.cards)
        self.count = -4 * self.num_decks  # Start with -4 for the four Aces


class Player:
    def __init__(self):
        self.hands = [[]]
        self.bets = [1]  # Initial bet for the first hand
        self.split_aces = [False]
        self.split_hands = [False]

    def take_card(self, card, hand_index=0):
        self.hands[hand_index].append(card)


def card_value(card):
    return 1 if card == 'A' else min(int(card) if card.isnumeric() else 10, 10)


def hand_value(hand):
    value = sum(card_value(card) for card in hand)
    num_aces = hand.count('A')
    while value <= 11 and num_aces:
        value += 10
        num_aces -= 1
    return value


def hand_value_low(hand):
    value = sum(card_value(card) for card in hand)
    return value


def basic_strategy(player, dealer_card, hand_index=0, count=None):
    # Enhanced basic strategy using the provided chart
    player_hand = player.hands[hand_index]
    player_value = hand_value(player_hand)  # Best value of hand (counts aces as 11 if possible without busting)
    dealer_value = card_value(dealer_card)
    is_soft = 'A' in player_hand and hand_value_low(player_hand) < 11

    # Check for pair splitting first
    if len(player_hand) == 2 and player_hand[0] == player_hand[1]:
        pair = player_hand[0]
        if (pair in ['A', '8'] and dealer_value != 1) or (pair == '9' and dealer_value not in [7, 10, 1]):
            return 'P'  # Always split Aces and Eights, and sometimes Nines
        elif pair in ['2', '3', '6', '7', '9'] and dealer_value < 6:
            return 'P'  # Split these pairs against lower dealer cards
        elif pair in ['2', '3', '7'] and dealer_value in [7]:
            return 'P'
        elif pair == '9' and dealer_value in [8, 9]:
            return 'P'
        elif pair == '4' and dealer_value in [5, 6]:
            return 'P'

    # Check for soft totals
    if is_soft:
        if player_value in range(13, 20) and dealer_value == 6:
            return 'D'  # Double down on soft 13 through 19 when dealer has 5 or 6
        if player_value in range(13, 19) and dealer_value == 5:
            return 'D'
        if player_value in range(15, 19) and dealer_value == 4:
            return 'D'
        if player_value in range(17, 19) and dealer_value == 3:
            return 'D'
        if player_value == 18 and dealer_value == 2:
            return 'D'
        if player_value >= 20:
            return 'S'
        if player_value == 18 and dealer_value not in [9, 10, 1]:
            return 'S'
        return 'H'

    if player_value >= 17:
        return 'S'
    if player_value in range(13, 17) and dealer_value < 7:
        return 'S'
    if player_value == 12 and dealer_value in [4, 5, 6]:
        return 'S'
    if player_value == 11 and dealer_value != 1:
        return 'D'
    if player_value == 10 and dealer_value not in [1, 10]:
        return 'D'
    if player_value == 9 and dealer_value in range(3, 7):
        return 'D'
    return 'H'


def uston_ss_count(deck, card):
    deck.count += USTON_SS_VALUES[card]


def simulate_game():
    average_winnings_insurance = collections.defaultdict(float)
    average_winnings_no_insurance = collections.defaultdict(float)
    games_played = collections.defaultdict(int)
    deck = Deck()
    game_count = 0
    while True:
        game_count += 1
        if len(deck.cards) < 312 * 0.3:  # Reshuffle if 30% of the shoe remains
            deck.shuffle()

        player = Player()
        dealer_card = deck.deal()
        uston_ss_count(deck, dealer_card)
        winnings_without_insurance = 0
        winnings_with_insurance = 0
        insurance = 0
        count_value = deck.count

        for i, hand in enumerate(player.hands):
            hand.append(deck.deal())
            uston_ss_count(deck, hand[-1])
            hand.append(deck.deal())
            uston_ss_count(deck, hand[-1])

            if dealer_card == 'A':
                insurance += player.bets[i] / 2

        winnings_with_insurance -= insurance
        hand_index = 0
        while hand_index < len(player.hands):
            hand = player.hands[hand_index]
            action = basic_strategy(player, dealer_card, hand_index)
            while action in ['H', 'P'] and hand_value(hand) < 21 and not player.split_aces[hand_index]:
                if action == 'H':
                    player.take_card(deck.deal(), hand_index)
                    uston_ss_count(deck, player.hands[hand_index][-1])
                elif action == 'P':
                    player.split_hands[hand_index] = True
                    player.split_aces[hand_index] = player.hands[hand_index][0] == 'A'
                    player.bets.append(player.bets[hand_index])  # Match the original bet for the new hand
                    player.hands.append([player.hands[hand_index].pop()])
                    player.split_hands.append(True)
                    player.split_aces.append(player.hands[hand_index][0] == 'A')
                    player.take_card(deck.deal(), hand_index)
                    uston_ss_count(deck, player.hands[hand_index][-1])
                    player.take_card(deck.deal(), len(player.hands) - 1)
                    uston_ss_count(deck, player.hands[len(player.hands) - 1][-1])
                action = basic_strategy(player, dealer_card, hand_index)

            if action == 'D' and not player.split_aces[hand_index]:
                player.bets[hand_index] *= 2  # Double the bet for Double Down
                player.take_card(deck.deal(), hand_index)
                uston_ss_count(deck, player.hands[hand_index][-1])

            hand_index += 1  # Move to the next hand

        dealer_hand = [dealer_card, deck.deal()]
        uston_ss_count(deck, dealer_hand[-1])
        while hand_value(dealer_hand) < 17:
            dealer_hand.append(deck.deal())
            uston_ss_count(deck, dealer_hand[-1])

        dealer_value = hand_value(dealer_hand)
        dealer_blackjack = dealer_value == 21 and len(dealer_hand) == 2
        winnings = 0
        for i, hand in enumerate(player.hands):
            player_value = hand_value(hand)
            player_blackjack = player_value == 21 and len(hand) == 2 and not player.split_hands[i]
            if player_value > 21:
                winnings -= player.bets[i]
            elif dealer_blackjack and not player_blackjack:
                winnings -= player.bets[i]
            elif player_blackjack and not dealer_blackjack:
                winnings += 1.5 * player.bets[i]  # Blackjack pays 3:2
            elif player_value > dealer_value or dealer_value > 21:
                winnings += player.bets[i]
            elif player_value < dealer_value:
                winnings -= player.bets[i]

        winnings_without_insurance += winnings
        winnings_with_insurance += winnings

        games_played[count_value] += 1
        if dealer_blackjack:
            winnings_with_insurance += insurance * 3

        average_winnings_insurance[count_value] = (average_winnings_insurance[count_value] * (
                    games_played[count_value] - 1) + winnings_with_insurance) / games_played[count_value]
        average_winnings_no_insurance[count_value] = (average_winnings_no_insurance[count_value] * (
                    games_played[count_value] - 1) + winnings_without_insurance) / games_played[count_value]

        if game_count % 1000000 == 0:
            save_to_csv(average_winnings_insurance, 'average_winnings_insurance.csv')
            save_to_csv(average_winnings_no_insurance, 'average_winnings_no_insurance.csv')
            save_difference_to_csv(average_winnings_insurance, average_winnings_no_insurance,
                                   'difference_in_winnings.csv')
            save_to_csv(games_played, 'games_played.csv')
            print(f'Saved game data after {game_count} games.')


if __name__ == "__main__":
    simulate_game()
