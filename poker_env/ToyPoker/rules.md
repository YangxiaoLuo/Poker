## initial setting
* deck:
  * rank: 234567
  * suit: 4 (scdh)
* player_num: 2 (p1, p2)
* small_bet: 1
* big_bet: 2
## pre_round
* chance action: deal hand cards
* p1--small_bet, p2--big_bet
* initial pot: (1, 2)
## flop round
* chance action: deal 3 public cards
* possible history:
  * p1-fold **game over**
  * p1-call(+1), p2-check **round over**
  * p1-call(+1), p2-fold **game over**
  * p1-call(+1), p2-raise(+2), p1-call(+2) **round over**
  * p1-call(+1), p2-raise(+2), p1-raise(+4), p2-call(+2) **round over**
  * p1-call(+1), p2-raise(+2), p1-raise(+4), p2-fold **game over**
  * p1-call(+1), p2-raise(+2), p1-fold **game over**
  * p1-raise(+3), p2-fold **game over**
  * p1-raise(+3), p2-call(+2) **round over**
  * p1-raise(+3), p2-raise(+4), p1-call(+2) **round over**
  * p1-raise(+3), p2-raise(+4), p1-fold **game over**

## final round
* chance action: deal 2 public cards
* p1 first take action
* possible history:
  * p1-check, p2-check/fold **game over**
  * p1-check, p2-raise(+4), p1-call(+4)/fold **game over**
  * p1-check, p2-raise(+4), p1-raise(+8), p2-call(+4)/fold **game over**
  * p1-raise(+4), p2-call(+4)/fold **game over**
  * p1-raise(+4), p2-raise(+8), p1-call(+4)/fold **game over**
  * p1-fold **game over**

