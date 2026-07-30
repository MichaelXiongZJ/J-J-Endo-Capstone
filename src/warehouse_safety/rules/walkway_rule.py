"""Pedestrian walkway compliance rule."""

from dataclasses import dataclass
from numbers import Real
from typing import Sequence

import cv2
import numpy as np


@dataclass(frozen=True)
class PersonObservation:
    """A detected person's bounding box and optional tracking metadata."""

    bbox_xyxy: tuple[float, float, float, float]
    confidence: float = 1.0
    track_id: int | None = None


@dataclass(frozen=True)
class WalkwayAssessment:
    """The walkway compliance result for one person."""

    track_id: int | None
    foot_point: tuple[float, float]
    is_inside: bool
    is_violation: bool


class WalkwayRule:
    """Assess whether a person's foot point lies within a walkway polygon."""

    def __init__(self, walkway_polygon: Sequence[tuple[float, float]]) -> None:
        """Initialize the rule with one walkway polygon."""
        points = tuple(walkway_polygon)
        if len(points) < 3:
            raise ValueError("walkway_polygon must contain at least three points.")

        validated_points: list[tuple[float, float]] = []
        for index, point in enumerate(points):
            if len(point) != 2:
                raise ValueError(
                    f"walkway_polygon point at index {index} must contain "
                    "exactly two coordinates."
                )

            x, y = point
            if not self._is_numeric(x) or not self._is_numeric(y):
                raise TypeError(
                    f"walkway_polygon coordinates at index {index} must be numeric."
                )
            validated_points.append((float(x), float(y)))

        self.walkway_polygon = tuple(validated_points)
        self._polygon_array = np.asarray(self.walkway_polygon, dtype=np.float32)

    @staticmethod
    def _is_numeric(value: object) -> bool:
        """Return whether a value is a non-boolean real number."""
        return isinstance(value, Real) and not isinstance(value, bool)

    @staticmethod
    def calculate_foot_point(
        bbox_xyxy: tuple[float, float, float, float],
    ) -> tuple[float, float]:
        """Return the bottom-center point of a valid bounding box."""
        if len(bbox_xyxy) != 4:
            raise ValueError("bbox_xyxy must contain exactly four coordinates.")

        if any(not WalkwayRule._is_numeric(value) for value in bbox_xyxy):
            raise TypeError("bbox_xyxy coordinates must be numeric.")

        x1, y1, x2, y2 = bbox_xyxy
        if x2 <= x1:
            raise ValueError("bbox_xyxy must satisfy x2 > x1.")
        if y2 <= y1:
            raise ValueError("bbox_xyxy must satisfy y2 > y1.")

        return ((float(x1) + float(x2)) / 2.0, float(y2))

    def assess_person(self, person: PersonObservation) -> WalkwayAssessment:
        """Assess one person against the walkway polygon."""
        foot_point = self.calculate_foot_point(person.bbox_xyxy)
        is_inside = (
            cv2.pointPolygonTest(self._polygon_array, foot_point, False) >= 0
        )

        return WalkwayAssessment(
            track_id=person.track_id,
            foot_point=foot_point,
            is_inside=is_inside,
            is_violation=not is_inside,
        )
