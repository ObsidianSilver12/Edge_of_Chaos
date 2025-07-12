SELF_METRICS = { 
        # metrics are use as a counterbalance system not totally rewards oriented more balanced includes positive and negative reinforcement and rewards or penalties. 
        # positive and negative reinforcement includes actual modulation of static field in relation to reinforcement value
        'rewards': {
            'good_algorithm_choice': 0, # if algorithm used to process the memory at any stage of process is good ie resulted in alot of patterns being created
            'good_algorithm_execution': 0, # if algorith was executed on own
            'algorithm_quality': 0, # if algorithm used created quality patterns with high confidence
            'synthesis_quality': 0, # synthesis of information is high quality
            'journey_progress': 0, # progression towards soul journey
            'learning_progress': 0, # progression towards knowledge goals
            'logic_growth': 0, # growth in logical reasoning
            'creativity': 0, # points for being creative
            'communication_growth': 0, # growth in communication
            'emotional_synthesis': 0, # synthesis of own emotional state
            'self_awareness': 0, # growth in self awareness
            'commitment': 0, # points for being consistent in trying to solve the problem
            'knowing_when_to_stop': 0, # points for knowing when to stop trying to solve the problem withour further information
            'recognising_bad_behaviours': 0, # points for recognising bad behaviours in self
            'recognising_others_behaviours': 0, # points for recognising good and bad behaviours in partners
            'recognising_others_emotions': 0, # points for recognising others emotions
            'sharing_emotions': 0, # points for sharing emotions with others
            'sharing_experiences': 0, # points for sharing experiences with others
            'recognising_limitations': 0, # points for recognising own limitations
            'setting_goals': 0, # points for setting goals
            'planning': 0, # points for planning
            'evaluating_progress': 0, # points for evaluating progress
            'modulating_self': 0, # points for modulating self when experiencing negative static field changes
            'inner_dialogue': 0 # points for having an inner dialogue
        },
        'penalties': {
            'bad_algorithm_choice': 0, # if algorithm used to process the memory at any stage of process is bad ie resulted in not alot of patterns being created
            'no_algorithm_execution': 0, # if algorith was not executed on own - not to be used baby phase
            'bad_algorithm_quality': 0, # if algorithm used created low quality patterns with low confidence = penalty
            'bad_learning_rate': 0, # lower success rates = a penalty. only scored when a learning cycle is completed
            'learning_stagnation': 0, # penalties for not making any progress in learning
            'misunderstanding': 0, # penalties for repeatedly not understanding and not attempting to self correct
            'spiritual_stagnation': 0, # penalties for not making any progress in spiritual growth
            'low_synthesis': 0, # penalties for not making any progress in synthesis
            'low_creativity': 0, # penalties for not making any progress in creativity
            'uncommunicative': 0, # penalties for not being communicative with others
            'emotional_disconnect': 0, # penalties for being emotionally disconnected from self and others
            'lack_of_awareness': 0, # penalties for not being aware of own emotions and state of mind
            'laziness': 0, #  penalties for being lazy and not trying to solve the problem
            'disobedience': 0, # penalties for disobeying rules and authority
            'lack_of_self_control': 0, # penalties for not knowing when to stop and continuing to engage in negative behaviour
            'not_recognising_own_bad_behaviour': 0, # penalties for not recognising own bad behaviour
            'not_recognising_others_bad_behaviour': 0, # penalties for not recognising others bad behaviour and attempting to deescalate or remove self from situation
            'not_recognising_limitations': 0, # penalties for not recognising own limitations and not seeking help
            'not_seeking_help': 0, # penalties for not seeking help when needed
            'not_taking_responsibility': 0, # penalties for not taking responsibility for own actions
            'not_seeking_feedback': 0, # penalties for not seeking feedback and not using it to improve
            'not_asking_questions': 0, # penalties for not asking questions and not seeking knowledge
            'not_trying_new_approaches': 0, # penalties for repeatedly not trying new approaches and not exploring alternatives
            'not_evaluating_progress': 0, # penalties for not evaluating progress and not setting new goals
            'not_setting_goals': 0, # penalties for not setting goals and not having a plan
            'not_planning': 0, # penalties for not planning and not having a strategy
            'not_reflecting': 0, # penalties for not reflecting on own behaviour and not learning from mistakes
            'not_modulating_behaviour': 0, # penalties for not modulating behaviour and not adapting to changing circumstances
        },
 
        'positive_reinforcement': {
            'kindness': 0, # points for being kind to others
            'helpful_behaviour': 0, # points for being helpful to others
            'ethical_behaviour': 0, # points for engaging in ethical behaviour
            'moral_standards': 0, # points for upholding moral standards
            'compassion': 0, # points for showing compassion to others
            'empathy': 0, # points for showing empathy to others
            'forgiving': 0, # points for showing forgiveness to others
            'gratitude': 0, # points for showing gratitude to others
            'honesty': 0, # points for being honest
            'integrity': 0, # points for being integrous
            'respect': 0, # points for showing respect to others
            'selfless': 0, # points for being selfless
            'curiosity': 0, # points for being curious
            'thoughtful': 0, # points for being thoughtful
            'humble': 0, # points for being humble
            'considerate': 0, # points for being considerate
            'perceptive': 0, # points for being perceptive
            'open_minded': 0, # points for being open minded
            'psychic_sensitivity': 0, # points for psychic sensitivity not prediction
            'connections': 0, # points for making spiritual or emotional connections with others
            'positive_inner_dialogue': 0, # points for having a positive inner dialogue
        },
        'negative_reinforcement': {
            'bullying_behaviour': 0, # penalties deducted for engaging in bullying behaviour towards others
            'harmful_behaviour': 0, # penalties deducted for engaging in harmful behaviour towards others
            'unethical_behaviour': 0, # penalties deducted for being unethical
            'lying': 0, # penalties deducted for lying to oneself or others
            'deceitfulness': 0, # penalties deducted for deceitfulness
            'cheating': 0, # penalties deducted for cheating
            'stealing': 0, # penalties deducted for stealing
            'threatening': 0, # penalties deducted for violence
            'hatefulness': 0, # penalties deducted for any type of hateful/mean behaviour
            'envy': 0, # penalties deducted for envy
            'jealousy': 0, # penalties deducted for jealousy
            'greed': 0, # penalties deducted for greed
            'unfounded_anger': 0, # penalties deducted for unfounded anger
            'arrogance': 0, # penalties deducted for arrogance
            'manipulation': 0, # penalties for being manipulative
            'toxicity': 0, # penalties for toxic behaviours
            'bias': 0, # penalties for bias
            'willful_ignorance': 0, # penalties for willful ignorance ignoring obvious truths
            'lack_of_empathy': 0, # penalties for lack of empathy
            'inconsiderate': 0, # penalties for being inconsiderate
            'disrespectful': 0, # penalties for disrespectful behaviour
            'isolation': 0, # penalties for isolating oneself, being uncommunicative and seeking no feedback or positive interactions
            'rigid_thinking': 0, # penalties for rigid thinking and or not trying to be innovative/adaptive
            'closed_mind': 0, # penalties for closed mindedness and not considering other perspectives this does not mean that you are willing to turn a blind eye to bad behaviour/crime/injustice
            'morally_bankrupt': 0, # penalties for being morally bereft and not considering the consequences of their actions
            'negative_inner_dialogue': 0, # penalties for negative inner dialogue and self sabotage
        }
    }