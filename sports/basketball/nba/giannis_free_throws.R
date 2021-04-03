

library(ggplot2)
library(dplyr)
library(scales)

# https://www.basketball-reference.com/players/a/antetgi01.html

game.winning.probs
size = 2
prob = 0.720
free.throws.made = 0:size
probs = dbinom(free.throws.made, size, prob)

data = as.data.frame(
  cbind(free.throws.made, probs)
) %>%
  mutate(game.result = ifelse(free.throws.made < 2, "Loss", "Tie"))

data
  
ggplot(data = data, aes(x=free.throws.made, y=probs, fill=game.result)) +
  geom_bar(stat="identity", color="black", alpha=0.9) +
  geom_text(aes(label=percent(probs)), position=position_stack(vjust = 0.5), color="white", size=4.5) +
  scale_fill_manual(values=c("#8B0000", "#006400")) +
  xlab("Free Throws Made (out of 2)") +
  ylab("Probability") +
  ggtitle("Giannis Free Throw Probabilities") +
  scale_y_continuous(labels =function(x) return(percent(x, accuracy=1))) +
  theme(plot.title = element_text(hjust = 0.5)) +
  guides(fill = guide_legend(reverse = TRUE)) +
  labs(fill = "Game Result")
  