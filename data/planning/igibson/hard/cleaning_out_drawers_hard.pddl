(define (problem cleaning_out_drawers_0)
    (:domain igibson)

    (:objects
     	bowl_1 bowl_2 - movable
    	cabinet_1 - container
        piece_of_cloth_1 - movable
    	sink_1 - object
    )
    
    (:init 
        (inside bowl_1 cabinet_1) 
        (inside bowl_2 cabinet_1) 
        (inside piece_of_cloth_1 cabinet_1) 
        (not (open cabinet_1))
    )
    
    (:goal 
        (and 
            (ontop bowl_1 sink_1) 
            (ontop bowl_2 sink_1) 
            (ontop piece_of_cloth_1 sink_1) 
        )
    )
)