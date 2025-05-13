(define (problem sorting_groceries_0)
    (:domain igibson)

    (:objects
        apple_1 apple_2 - movable
        orange_1 orange_2 - movable
        electric_refrigerator_1 - container
        table_1 - object
    )

    (:init
    )

    (:goal
        (and
            (inside apple_1 electric_refrigerator_1)
            (ontop orange_2 table_1)
            (nextto orange_1 orange_2)

        )
    )
)
