import pytest

from warehouse_safety.rules.walkway_rule import PersonObservation, WalkwayRule


@pytest.fixture
def walkway_rule() -> WalkwayRule:
    polygon = [(100, 100), (500, 100), (500, 500), (100, 500)]
    return WalkwayRule(polygon)


def test_person_inside_walkway(walkway_rule: WalkwayRule) -> None:
    person = PersonObservation(bbox_xyxy=(200, 150, 300, 400))

    assessment = walkway_rule.assess_person(person)

    assert assessment.foot_point == (250.0, 400.0)
    assert assessment.is_inside is True
    assert assessment.is_violation is False


def test_person_outside_walkway(walkway_rule: WalkwayRule) -> None:
    person = PersonObservation(bbox_xyxy=(550, 150, 650, 400))

    assessment = walkway_rule.assess_person(person)

    assert assessment.is_inside is False
    assert assessment.is_violation is True


def test_person_on_walkway_boundary_is_inside(
    walkway_rule: WalkwayRule,
) -> None:
    person = PersonObservation(bbox_xyxy=(200, 150, 400, 500))

    assessment = walkway_rule.assess_person(person)

    assert assessment.foot_point == (300.0, 500.0)
    assert assessment.is_inside is True
    assert assessment.is_violation is False


def test_polygon_requires_at_least_three_points() -> None:
    with pytest.raises(
        ValueError,
        match="walkway_polygon must contain at least three points",
    ):
        WalkwayRule([(100, 100), (500, 100)])


def test_polygon_coordinates_must_be_numeric() -> None:
    polygon = [(100, 100), (500, "invalid"), (100, 500)]

    with pytest.raises(TypeError, match="coordinates at index 1 must be numeric"):
        WalkwayRule(polygon)  # type: ignore[arg-type]


@pytest.mark.parametrize("x2", [100, 99])
def test_bounding_box_requires_x2_greater_than_x1(
    walkway_rule: WalkwayRule,
    x2: float,
) -> None:
    person = PersonObservation(bbox_xyxy=(100, 100, x2, 200))

    with pytest.raises(ValueError, match="x2 > x1"):
        walkway_rule.assess_person(person)


@pytest.mark.parametrize("y2", [100, 99])
def test_bounding_box_requires_y2_greater_than_y1(
    walkway_rule: WalkwayRule,
    y2: float,
) -> None:
    person = PersonObservation(bbox_xyxy=(100, 100, 200, y2))

    with pytest.raises(ValueError, match="y2 > y1"):
        walkway_rule.assess_person(person)


def test_bounding_box_coordinates_must_be_numeric(
    walkway_rule: WalkwayRule,
) -> None:
    person = PersonObservation(
        bbox_xyxy=(100, 100, "invalid", 200),  # type: ignore[arg-type]
    )

    with pytest.raises(TypeError, match="bbox_xyxy coordinates must be numeric"):
        walkway_rule.assess_person(person)


def test_track_id_is_preserved(walkway_rule: WalkwayRule) -> None:
    person = PersonObservation(
        bbox_xyxy=(200, 150, 300, 400),
        track_id=42,
    )

    assessment = walkway_rule.assess_person(person)

    assert assessment.track_id == 42
