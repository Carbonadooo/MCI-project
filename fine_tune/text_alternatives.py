"""
Text alternatives for data augmentation

Each action class has 11 text descriptions (1 canonical + 10 variations)
Used during training to increase text diversity
"""

TEXT_ALTERNATIVES = {
    'brush_teeth': [
        'brush teeth', 'brushing my teeth', 'cleaning teeth with toothbrush',
        'doing tooth brushing', 'brushing dental surface', 'tooth-cleaning motion',
        'hygiene brushing routine', 'scrubbing teeth', 'brushing oral area',
        'morning brushing motion', 'performing tooth hygiene'
    ],
    'clap_once': [
        'clap once', 'single clap', 'clap one time', 'quick clap',
        'one-hand clap motion', 'short single clapping', 'do one clap',
        'clap briefly', 'a single hand clap', 'quick single applaud', 'fast one-time clap'
    ],
    'cut_vegetables_with_knife': [
        'cut vegetables with knife', 'chopping vegetables', 'slicing veggies with a knife',
        'cutting vegetables on board', 'veggie chopping motion', 'knife cutting vegetables',
        'preparing vegetables by cutting', 'slicing produce', 'chopping food items',
        'dicing vegetables', 'performing knife-cutting on vegetables'
    ],
    'drink_water': [
        'drink water', 'taking a sip of water', 'drinking from a cup', 'sipping water',
        'taking a drink', 'consuming water', 'hydrating with water', 'gulping water',
        'lifting cup to drink', 'drinking from bottle', 'taking a water gulp'
    ],
    'flick_switch_up_down': [
        'flick switch up down', 'flipping a switch', 'toggling switch up and down',
        'switching on/off', 'flicking the light switch', 'toggling the lever',
        'quick switch flick', 'flipping electrical switch', 'moving switch upward/downward',
        'performing switch toggle motion', 'pressing switch up and down'
    ],
    'fold_clothes': [
        'fold clothes', 'folding laundry', 'folding a shirt/pants', 'doing clothes folding',
        'tidying clothes by folding', 'garment folding motion', 'folding fabric items',
        'organizing clothes', 'folding garments neatly', 'performing laundry folding',
        'folding clothing items'
    ],
    'high_five_motion': [
        'high five motion', 'giving a high-five', 'raising hand for high-five',
        'high-five gesture', 'performing high-five action', 'slapping palms together',
        'friendly high-five', 'reaching out for high-five', 'celebratory high-five motion',
        'hand-to-hand high-five', 'quick high-five tap'
    ],
    'mimimi': [
        'mimimi', 'miming "mi-mi-mi"', 'making "mimimi" sound', 'repetitive "mi" vocal motion',
        'lip-moving mi-mi-mi gesture', 'doing mimimi mouth shape', 'small vocalization motion',
        'repeating syllable "mi"', 'low-intensity vocal gesture', 'rhythmic "mi mi mi" motion',
        'mimicking "mimi" sound pattern'
    ],
    'open_close_drawer': [
        'open close drawer', 'opening and shutting a drawer', 'pulling drawer out and pushing in',
        'drawer open-close motion', 'sliding drawer in/out', 'opening drawer then closing it',
        'drawer sliding motion', 'drawer pull-and-push action', 'accessing a drawer briefly',
        'toggling drawer open/closed', 'moving drawer in and out'
    ],
    'open_close_notebook': [
        'open close notebook', 'opening and shutting a notebook', 'flipping notebook open and closed',
        'notebook open-close gesture', 'open book then close it', 'flipping notebook cover',
        'notebook cover motion', 'opening notebook then closing', 'toggling notebook open/shut',
        'flipping notebook pages open/closed', 'performing notebook open-close action'
    ],
    'open_door_with_handle': [
        'open door with handle', 'opening a door', 'pulling door handle', 'door opening motion',
        'turning handle and opening', 'accessing through door', 'handle turning motion',
        'opening entry door', 'pulling door open', 'door handle operation', 'entering through door'
    ],
    'open_refrigerator': [
        'open refrigerator', 'opening fridge door', 'pulling refrigerator open',
        'accessing refrigerator', 'fridge door opening', 'opening the fridge',
        'pulling fridge handle', 'refrigerator access motion', 'opening cold storage',
        'fridge opening gesture', 'accessing food storage'
    ],
    'pick_and_place': [
        'pick and place', 'picking up and putting down', 'grabbing and placing object',
        'lift and place motion', 'object transfer action', 'pick up then set down',
        'grabbing and releasing object', 'moving object from A to B', 'object relocation',
        'lifting and positioning item', 'transferring object motion'
    ],
    'pour_water_into_cup': [
        'pour water into cup', 'pouring water', 'filling cup with water',
        'water pouring motion', 'pouring liquid into container', 'filling a cup',
        'transferring water to cup', 'pouring beverage', 'cup filling action',
        'liquid pouring gesture', 'water transfer motion'
    ],
    'punch_forward': [
        'punch forward', 'throwing a punch', 'forward punching motion', 'straight punch',
        'punching ahead', 'forward strike motion', 'extending fist forward',
        'performing forward punch', 'straight arm punch', 'forward jabbing motion',
        'front punch action'
    ],
    'screw_bottle_cap': [
        'screw bottle cap', 'twisting bottle cap', 'screwing cap on bottle',
        'tightening bottle cap', 'rotating cap motion', 'closing bottle with cap',
        'bottle cap screwing motion', 'twisting cap closed', 'cap tightening action',
        'screwing lid onto bottle', 'bottle sealing motion'
    ],
    'shake_bottle': [
        'shake bottle', 'shaking a bottle', 'bottle shaking motion', 'agitating bottle',
        'vigorous bottle shake', 'shaking container', 'bottle mixing motion',
        'rapid bottle movement', 'shaking liquid container', 'bottle agitation',
        'performing bottle shake'
    ],
    'squeeze_hand_sanitizer': [
        'squeeze hand sanitizer', 'dispensing hand sanitizer', 'pumping sanitizer',
        'squeezing sanitizer bottle', 'applying hand sanitizer', 'sanitizer dispensing motion',
        'pressing sanitizer pump', 'getting hand sanitizer', 'squeezing hygiene gel',
        'dispensing cleaning gel', 'sanitizer application motion'
    ],
    'stir_with_spoon': [
        'stir with spoon', 'stirring with a spoon', 'mixing with spoon', 'spoon stirring motion',
        'circular stirring action', 'mixing contents with spoon', 'rotating spoon in liquid',
        'stirring mixture', 'performing stirring motion', 'spoon mixing action',
        'circular spoon movement'
    ],
    'throw_small_object': [
        'throw small object', 'tossing a small item', 'throwing object forward',
        'small object toss', 'lobbing small item', 'throwing something small',
        'object throwing motion', 'tossing item away', 'small throw action',
        'underhand object toss', 'discarding by throwing'
    ],
    'twist_towel': [
        'twist towel', 'twisting a towel', 'wringing towel', 'towel twisting motion',
        'rotating towel fabric', 'twisting cloth', 'wringing out towel',
        'towel wringing action', 'rotating towel ends', 'performing towel twist',
        'squeezing towel by twisting'
    ],
    'unplug_usb_cable': [
        'unplug usb cable', 'removing USB cable', 'unplugging USB', 'pulling out USB cable',
        'disconnecting USB', 'USB cable removal', 'extracting USB connector',
        'unplugging cable', 'removing USB connection', 'pulling USB out',
        'USB disconnection motion'
    ],
    'use_hammer': [
        'use hammer', 'hammering motion', 'using a hammer', 'striking with hammer',
        'hammer hitting motion', 'performing hammer action', 'hammering downward',
        'tool hammering motion', 'hammer strike action', 'pounding with hammer',
        'downward hammer motion'
    ],
    'wave_hand_left_right': [
        'wave hand left right', 'waving hand', 'hand waving motion', 'left-right hand wave',
        'greeting wave gesture', 'waving hello', 'horizontal hand wave',
        'side-to-side hand motion', 'friendly wave gesture', 'waving hand back and forth',
        'hand oscillation motion'
    ],
    'wipe_table_back_forth': [
        'wipe table back forth', 'wiping table surface', 'cleaning table', 'back and forth wiping',
        'table cleaning motion', 'surface wiping action', 'wiping back and forth',
        'table surface cleaning', 'horizontal wiping motion', 'cleaning table top',
        'back-forth cleaning gesture'
    ]
}

