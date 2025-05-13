(define (problem organizing_boxes_in_garage_0)
    (:domain igibson)

    (:objects
        ball_1 ball_2 plate_1 plate_2 plate_3 saucepan_1 - movable
        shelf_1 cabinet_1 - container
        carton_1 carton_2 - container
    )

    (:init
        (inside plate_1 shelf_1) 
    )

    (:goal 
        (and
            (inside ball_1 carton_1)
            (inside plate_1 carton_1)
        )
    )
)
