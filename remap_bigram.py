

class RemapBigram:
    def __init__(self):
        self.remap = {}
        self.reverse_map = {}
        pass

    def create_remap(self):
        # implement the remap
        pass
    
    def get_unique(self):
        return ""
    
    def get_reversed_map(self):
        if self.reverse_map == {}:
            self.create_remap()
            log.error("Bad use of get reversed map")
        return self.reverse_map

    def get_map(self):
        if self.remap == {}:
            self.create_remap()
        return self.remap

    def get_potential_original_tokens(self):
        """
        Attacker needs to implement this function (instead of using get_map and
        get_remap)
        """
        pass
    
    # we assume that the vocab is id: 'word'
    def remap_input_ids(self, input_ids, attention_mask, name=""):
        """
        map tokens to new tokens
        """
        survived_tokens = 0
        total_tokens = 0 
        if self.remap == {}:
            self.create_remap()
        new_input_ids = input_ids
        # cpy = input_ids
        for i, ids in enumerate(input_ids):
            for j, token in enumerate(ids):
                if token == 0:
                    continue
                if int(token) == self.remap[int(token)]:
                    survived_tokens = survived_tokens + 1
                new_input_ids[i][j] = self.remap[int(token)]
                total_tokens = total_tokens + 1
        print("survived tokens:", survived_tokens, survived_tokens/ total_tokens)
        return new_input_ids
    
    def remove_forbidden_tokens(self, indices_to_shuffle, forbid):
        if forbid:
            a_file = open("roberta_gpt_mapper.pkl", "rb")
            roberta_gpt_mapper = pickle.load(a_file)
            forbidden = []
            for i in range(len(roberta_gpt_mapper)):
                if roberta_gpt_mapper[i] == -1:
                    forbidden.append(i)
                    self.remap[i] = i
            forbidden.sort(reverse=True)
            for rm in forbidden:
                indices_to_shuffle.pop(rm)
            a_file.close()
        else:
            indices_to_shuffle.pop(50264)
            indices_to_shuffle.pop(2)
            indices_to_shuffle.pop(1)
            indices_to_shuffle.pop(0)
            self.remap[2] = 2
            self.remap[50264] = 50264
            self.remap[1] = 1
            self.remap[0] = 0
