class OneKLBClub:
    TOTAL = 1000
    def __init__(self, deadlift=None, squat=None, benchpress=None):
        if sum(bool(x) for x in (deadlift, squat, benchpress)) != 2:
            raise ValueError('Must enter values for only two of deadlift, squat and benchpress arguments')
        self.deadlift = deadlift or OneKLBClub.TOTAL - (squat + benchpress)
        self.squat = squat or OneKLBClub.TOTAL - (deadlift + benchpress)
        self.benchpress = benchpress or OneKLBClub.TOTAL - (deadlift + squat)

        assert self.deadlift + self.squat + self.benchpress == 1000

    def __repr__(self):
        return f'OneKLBClub(deadlift={self.deadlift}, squat={self.squat}, benchpress={self.benchpress})'


if __name__ == '__main__':
    attempt = OneKLBClub(deadlift=425, squat=290)
    print(attempt)



