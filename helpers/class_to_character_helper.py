

class ClassToCharacterHelper:

    character_map = {
        0: 'Gary',
        1: 'Mr. Krabs',
        2: 'Patrick Star',
        3: 'Plankton',
        4: 'Sandy Cheeks',
        5: 'Spongebob Squarepants',
        6: 'Squidward'
    }

    def map_class_to_character_name(self, class_number):
        return self.character_map[class_number]
