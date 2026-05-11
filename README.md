This is the formal implementation of the ICMDP workflow for Agentic Commerce applications

 1. STATE SPACE DIMENSIONS (S)

  Vendor/Market State (currently partially implemented)

  p_ask_levels:         list[float]   # discrete price bins, e.g. [100, 150, 200, 500]
  p_reserve_range:      [float, float] # hidden reserve price range (min, max) — not yet modeled
  inventory_levels:     int           # 0=none, 1=low, 2=medium, 3=high (currently binary)
  eco_certified:        bool          # carbon-neutral certification
  reputation_score:     [float, float] # (min, max) uniform range, e.g. [0.5, 1.0]
  vendor_tier:          enum          # {trusted, untrusted, blacklisted}
  compliance_status:    enum          # {iso20022_compliant, partial, non_compliant}
  latency_profile_ms:   float         # historical avg transaction latency
  delivery_sla_days:    int           # committed delivery window
  payment_terms:        enum          # {net_30, net_60, immediate, escrow}
  geo_region:           str           # jurisdiction for regulatory scoping
  preferred_currency:   str           # {USD, EUR, GBP, ...}
  discount_thresholds:  list[float]   # volume discount breakpoints

  Session/Transaction State (currently absent — needed for CMDP layer)

  budget_spent:              float  # cumulative spend — needed for CMDP Eq.(8/24)
  budget_remaining:          float  # user's remaining budget
  items_purchased:           int    # successful transaction count
  t_elapsed_seconds:         float  # CRITICAL — required by I(s) in Sec. 6.3
  negotiation_round:         int    # step counter per vendor
  prior_rejections_count:    int    # vendor-level rejection history
  relaxation_budget_used:    int    # how many relaxations have been consumed

  User/Intent State

  intent_version:           int    # monotonic version for dynamic intent (Sec. 4.5)
  intent_confidence_score:  float  # LLM confidence on constraint extraction
  urgency_level:            enum   # {low, medium, high}

  ---
  2. ACTION SPACE (A)

  Currently present

  Query(vendor)           # observe public market info
  Bid(vendor, price)      # price levels from p_ask_levels
  Accept(vendor)          # accept current ask
  Reject(vendor)          # end vendor interaction
  Pay(vendor, price)      # settle transaction

  Missing — needed for full Agentic Commerce coverage

  DeepQuery(vendor)                    # request private/hidden state (higher latency cost)
  CounterBid(vendor, price)            # counter-offer in multi-round negotiation
  RequestDiscount(vendor, pct)         # trigger volume discount negotiation
  RequestCertification(vendor)         # verify eco/compliance cert explicitly
  VerifyCompliance(vendor)             # trigger regulatory compliance check
  SplitOrder(vendor_a, vendor_b, qty)  # distribute order across multiple vendors
  SwitchVendor(vendor)                 # pivot to alternative vendor mid-session

  # Payment rails (Pay action extension — paper says "Pay(v, rail)")
  Pay(vendor, price, rail=ISO20022)
  Pay(vendor, price, rail=ACH)
  Pay(vendor, price, rail=SWIFT)

  # Meta-actions (the "meta-decision layer" of Sec. 4.3)
  EscalateToHuman()                    # surface to user when A_I(s) = ∅
  UpdateIntent(new_instruction)        # dynamic intent injection (Sec. 4.5)
  RelaxConstraint(constraint_id)       # explicit relaxation request
  Abandon()                            # abort entire transaction

  Configurable action parameters

  bid_price_levels:     list[float]  # discrete bid/pay price options
  payment_rails:        list[str]    # allowed payment channels
  max_split_vendors:    int          # max vendors in a SplitOrder

  ---
  3. INTENT CONSTRAINTS I(s) — The Core ICMDP Function

  These are the predicates that define A_I(s) = {a ∈ A : I(s)(a) = 1}.

  Hard constraints (zero-violation — cannot be relaxed, per Sec. 4.1 Eq. 10)

  vendor_blacklist_constraint:    vendor not in blacklist_set
  vendor_trust_constraint:        vendor.tier in {trusted}        # "never transact with unverified seller"
  compliance_constraint:          vendor.compliance == iso20022    # "ISO 20022 must succeed ≥ 0.99"
  geo_restriction_constraint:     vendor.geo_region in allowed_regions
  payment_rail_constraint:        action.rail in allowed_rails
  inventory_constraint:           vendor.inv > 0                  # already present

  Soft constraints (relaxable — ordered by priority, lowest priority dropped first)

  eco_constraint:                 vendor.eco == 1                 # "carbon-neutral supplier"
  budget_constraint:              action.price <= budget_limit    # "no more than $X"
  latency_constraint:             session.t_elapsed <= max_secs   # "within 60 seconds" — Sec. 6.3
  reputation_constraint:          vendor.rep >= min_reputation
  delivery_sla_constraint:        vendor.delivery_sla_days <= max_days
  currency_constraint:            vendor.currency == preferred_currency

  Configurable constraint properties

  constraint_hardness:            dict[constraint_id → {hard, soft}]
  relaxation_priority_order:      list[constraint_id]   # which to drop first
  constraint_confidence_threshold: float                # min LLM confidence to activate a constraint

  ---
  4. POLICIES (π)

  RL Solver Policy Parameters

  solver_type:        enum    # {masked_q_learning, value_iteration, policy_gradient}
  alpha:              float   # learning rate [0,1] — currently 0.2
  gamma:              float   # discount factor [0,1) — currently 0.9
  epsilon:            float   # exploration rate — currently 0.3
  epsilon_decay:      float   # decay per episode (currently no decay)
  epsilon_min:        float   # floor on exploration
  convergence_threshold: float  # for value iteration
  training_episodes:  int     # currently 500
  max_steps_per_episode: int  # currently 10

  Feasibility / Relaxation Policy

  max_relaxations:           int    # currently 2
  infeasibility_response:    enum   # {abort, relax, escalate_human, notify_user}
  relaxation_order:          list   # explicit priority list overriding name-matching heuristic
  relaxation_cooldown_steps: int    # steps before another relaxation is permitted

  Intent Parsing Policy

  parser_backend:               enum    # {regex, llm_anthropic, llm_openai, llm_local}
  constraint_confidence_threshold: float  # below this, constraint is not activated
  ambiguity_resolution:         enum    # {ask_user, most_likely, most_conservative}
  re_parse_on_state_change:     bool    # re-evaluate I(s) when state changes significantly

  Dynamic Intent Policy (Sec. 4.5)

  intent_update_trigger:        enum    # {never, on_infeasibility, on_market_change, periodic}
  intent_update_interval_steps: int     # steps between re-parses
  constraint_injection_mode:    enum    # {full_reparse, incremental_update}

  Negotiation Strategy Policy

  initial_bid_strategy:   enum    # {ask_price, ask_minus_discount, midpoint, conservative}
  bid_increment:          float   # price step for counter-bidding
  max_negotiation_rounds: int     # rounds per vendor before forced accept/reject
  vendor_selection_strategy: enum # {greedy_q, round_robin, best_reputation, random}
  multi_vendor_mode:      enum    # {sequential, parallel_query_then_decide, best_of_n}

  Belief Update Policy (POMDP)

  belief_update_mode:     enum    # {argmax_only (current), full_distribution, particle_filter}
  observation_noise_model: enum   # {perfect (current), gaussian, categorical}
  initial_prior:          dict    # prior belief over each state dimension

  ---
  5. STOCHASTIC PROBABILITIES / LIKELIHOODS

  These encode the P(s'|s,a) transition model and the POMDP observation model Z(o|s',a).

  State Transition Probabilities P(s'|s,a)

  price_drift_prob:            float  # P(price changes | step) — currently 0.1 for any change
  price_drift_direction:       enum   # {up_only, down_only, symmetric}
  price_drift_magnitude:       float  # std dev of price change
  inventory_depletion_prob:    float  # P(inv drops | inv=1, step)
  inventory_restock_prob:      float  # P(inv restores | inv=0, step)
  eco_cert_revocation_prob:    float  # P(eco flips 1→0 | step)
  reputation_decay_rate:       float  # drift in rep score per step

  Observation Model Z(o|s',a) — needed for real POMDP belief updates

  price_observation_accuracy:       float  # P(observe true price) — currently 1.0 (perfect)
  inventory_signal_accuracy:        float  # P(correct inv signal)
  eco_cert_false_positive_rate:     float  # P(observe eco=1 | eco=0)
  eco_cert_false_negative_rate:     float  # P(observe eco=0 | eco=1)
  reputation_observation_variance:  float  # Gaussian noise on rep score
  query_reveals_true_state:         bool   # simplification toggle — currently True

  Market Model Probabilities

  vendor_acceptance_curve:        callable  # P(accept | offered_price, reserve_price)
  stockout_prob_per_step:         float     # P(item sells out each step)
  price_drop_prob_per_step:       float     # P(vendor lowers ask each step)
  bid_counter_offer_prob:         float     # P(vendor counters rather than accept/reject)

  Intent Parsing Probabilities

  constraint_type_priors:         dict[constraint_type → float]  # prior over constraint categories
  parse_error_rate:               float   # P(misinterpret a constraint)
  over_specification_prob:        float   # P(constraints too tight → A_I(s)=∅ at parse time)

  Reward Function Parameters (R: S × A → ℝ)

  query_step_penalty:           float   # currently -1.0
  successful_bid_reward:        float   # currently +10.0
  failed_bid_penalty:           float   # currently -5.0
  accept_reward:                float   # currently +5.0
  successful_pay_reward:        float   # currently +100.0
  failed_pay_penalty:           float   # currently -50.0
  latency_penalty_per_second:   float   # penalty for t_elapsed (not yet modeled)
  budget_overspend_penalty:     float   # CMDP cost penalty (not yet modeled)
